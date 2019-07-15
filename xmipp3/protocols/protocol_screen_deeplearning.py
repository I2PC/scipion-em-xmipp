# **************************************************************************
# *
# * Authors:  Ruben Sanchez (rsanchez@cnb.csic.es), April 2017
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

import os, sys
import re

from pyworkflow import VERSION_2_0
from pyworkflow.utils.path import copyTree
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtProcessParticles
import pyworkflow.em.metadata as md

from xmipp3.convert import writeSetOfParticles, setXmippAttributes
from xmipp3.utils import validateDLtoolkit

N_MAX_NEG_SETS= 5

class XmippProtScreenDeepLearning(ProtProcessParticles):
    """ Protocol for screening particles using deep learning. """
    _label = 'screen deep learning'
    _lastUpdateVersion = VERSION_2_0

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        # GPU settings
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Use GPU (vs CPU)",
                       help="Set to true if you want to use GPU implementation")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")
        form.addParallelSection(threads=2, mpi=0)
        form.addSection(label='Input')
        form.addParam('doContinue', params.BooleanParam, default=False,
                      label='Use previously trained model?',
                      help='If you set to *Yes*, you should select a previous '
                           'run of type *%s* class and some of the input parameters '
                           'will be taken from it.' % self.getClassName())
        form.addParam('continueRun', params.PointerParam,
                      label='Select previous run', allowsNull=True,
                      condition='doContinue', pointerClass=self.getClassName(),
                      help='Select a previous run to continue from.')
        form.addParam('keepTraining', params.BooleanParam,
                      label='Continue training on previously trainedModel?',
                      default=True,condition='doContinue',
                      help='If you set to *Yes*, you should provide training set')
            
        form.addParam('inTrueSetOfParticles', params.PointerParam,
                      label="True particles", pointerClass='SetOfParticles',
                      allowsNull=True, condition="not doContinue or keepTraining",
                      help='Select a set of particles that contains mostly true particles')

        form.addParam('numberOfNegativeSets', params.IntParam,
                      label='Number of different negative dataset',
                      default='1', condition="not doContinue or keepTraining",
                      help='Data from all negative datasets will be used for training. '
                           'Maximun number is 4.\n')

        for num in range(1, N_MAX_NEG_SETS):
            form.addParam('negativeSet_%d' % num, params.PointerParam,
                          label="Set of negative train particles %d" % num,
                          condition='(numberOfNegativeSets<=0 or numberOfNegativeSets >=%d) '
                                    'and (not doContinue or keepTraining)' % num,
                          pointerClass='SetOfParticles', allowsNull=True,
                          help='Select the set of negative particles for training.')
                          
            form.addParam('inNegWeight_%d'%num, params.IntParam,
                          label="Weight of negative train particles %d" % num,
                          expertLevel=params.LEVEL_ADVANCED,
                          default='1', allowsNull=True,
                          condition='(numberOfNegativeSets<=0 or numberOfNegativeSets >=%d) and '
                                    '(not doContinue or keepTraining)'%num,
                          help='Select the weigth for the negative set of particles. '
                               'The weight value indicates the number of times '
                               'each image may be included at most per epoch. '
                               'Positive particles are weighted with 1. '
                               'If weight is -1, weight will be calculated such '
                               'that the contribution of additional data is '
                               'equal to the contribution of positive particles')

        form.addParam('predictSetOfParticles', params.PointerParam,
                      label="Set of putative particles to score",
                      pointerClass='SetOfParticles',
                      help='Select the set of putative particles to classify as good (score close '
                            'to 1.0) or bad (score close to 0.0).')

        form.addSection(label='Training')
        
        form.addParam('nEpochs', params.FloatParam,
                      label="Number of epochs", default=5.0,
                      condition="not doContinue or keepTraining",
                      help='Number of epochs for neural network training.')
        form.addParam('learningRate', params.FloatParam,
                      label="Learning rate", default=1e-4,
                      condition="not doContinue or keepTraining",
                      help='Learning rate for neural network training')
        form.addParam('auto_stopping',params.BooleanParam,
                      label='Auto stop training when convergency is detected?',
                      default=True, condition="not doContinue or keepTraining",
                      help='If you set to *Yes*, the program will automatically '
                           'stop training if there is no improvement for '
                           'consecutive 2 epochs, learning rate will be '
                           'decreased by a factor 10. '
                           'If learningRate_t < 0.01*learningrate_0 training will stop. '
                           'Warning: Sometimes convergency seems to be reached, '
                           'but after time, improvement can still happen. '
                           'Not recommended for very small data sets (<100 true particles)')
        form.addParam('l2RegStrength', params.FloatParam,
                      label="Regularization strength",
                      default=1e-5, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue or keepTraining",
                      help='L2 regularization for neural network weights.'
                           'Make it bigger if suffering overfitting. '
                           'Typical values range from 1e-1 to 1e-6')
        form.addParam('nModels', params.IntParam,
                      label="Number of models for ensemble",
                      default=2, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue",
                      help='Number of models to fit in order to build an ensamble. '
                           'Tipical values are 1 to 5. The more the better '
                           'until a point where no gain is obtained. '
                           'Each model increases running time linearly')
                           
        form.addParam('doTesting', params.BooleanParam, default=False,
                      label='Perform testing after training?', expertLevel=params.LEVEL_ADVANCED,
                      help='If you set to *Yes*, you should select a testing '
                           'positive set and a testing negative set')
        form.addParam('testPosSetOfParticles', params.PointerParam,
                      label="Set of positive test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles',condition='doTesting',
                      help='Select the set of ground true positive particles.')
        form.addParam('testNegSetOfParticles', params.PointerParam,
                      label="Set of negative test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles', condition='doTesting',
                      help='Select the set of ground false positive particles.')

    def _validate(self):
        return validateDLtoolkit()

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """            
        """
        def _getFname2WeightDict(fnameToSetAndWeight):
            """
            arg: fnameToSetAndWeight= { fname: [(SetOfParticles, weight:int)]}
            return:  { fname: { weight:int }

            """
            if fnameToSetAndWeight is None:
                return None
            dictONameToWeight = {fname: fnameToSetAndWeight[fname][-1]
                                   for fname in fnameToSetAndWeight
                                     if not fnameToSetAndWeight[fname][0] is None}

            if len(dictONameToWeight) == 0:
                return None
            else:
                return dictONameToWeight

        posSetTrainDict = {self._getExtraPath("inputTrueParticlesSet.xmd"): 1}
        
        negSetTrainDict = {}
        for num in range(1, N_MAX_NEG_SETS):
            if self.numberOfNegativeSets <= 0 or self.numberOfNegativeSets >= num:
                negSetTrainDict[self._getExtraPath("negativeSet_%d.xmd"%num)] = \
                                          self.__dict__["inNegWeight_%d"%num].get()
                    
        setPredictDict = {self._getExtraPath("predictSetOfParticles.xmd"): 1}

        if self.doTesting.get() and self.testPosSetOfParticles.get() and self.testNegSetOfParticles.get():
          setTestPosDict = {self._getExtraPath("testTrueParticlesSet.xmd"): 1}
          setTestNegDict = {self._getExtraPath("testFalseParticlesSet.xmd"): 1}
        else:
          setTestPosDict = None
          setTestNegDict = None
          
        self._insertFunctionStep('convertInputStep', posSetTrainDict,
                                 negSetTrainDict, setPredictDict,
                                 setTestPosDict, setTestNegDict)
        if not self.doContinue.get() or self.keepTraining.get():
          self._insertFunctionStep('train', posSetTrainDict, negSetTrainDict)
        self._insertFunctionStep('predict', setTestPosDict, setTestNegDict,setPredictDict)
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, *dataDicts):
        def __getSetOfParticlesFromFname(fname):
          if fname== self._getExtraPath("inputTrueParticlesSet.xmd"):
            return self.inTrueSetOfParticles.get()
          elif fname== self._getExtraPath("predictSetOfParticles.xmd"):
            return self.predictSetOfParticles.get()
          elif fname== self._getExtraPath("testTrueParticlesSet.xmd"):
            return self.testPosSetOfParticles.get()
          elif fname== self._getExtraPath("testFalseParticlesSet.xmd"):
            return self.testNegSetOfParticles.get()
          else:
            matchOjb= re.match( self._getExtraPath("negativeSet_(\d+).xmd"), fname)
            if matchOjb:
              num= matchOjb.group(1)
              return self.__dict__["negativeSet_%s"%num].get()
            else:
              raise ValueError("Error, unexpected fname")
                     
        if ((not self.doContinue.get() or self.keepTraining.get())
                and self.nEpochs.get() > 0):
            assert not self.inTrueSetOfParticles.get() is None, \
                    "Positive particles must be provided for training if nEpochs!=0"
                    
        for dataDict in dataDicts:
            if not dataDict is None:
                for fnameParticles in sorted(dataDict):
                    setOfParticles= __getSetOfParticlesFromFname(fnameParticles)
                    writeSetOfParticles(setOfParticles, fnameParticles)
                    
    def __dataDict_toStrs(self, dataDict):
        fnamesStr=[]
        weightsStr=[]
        for fname in dataDict:
          fnamesStr.append(fname)
          weightsStr.append(str(dataDict[fname]) )
        return ":".join(fnamesStr), ":".join(weightsStr)
        
    def train(self, posTrainDict, negTrainDict):
        """
            posTrainDict, negTrainDict: { fnameToMetadata: weight (int) }
        """
        nEpochs = self.nEpochs.get()
        netDataPath = self._getExtraPath('nnetData')
        if self.doContinue.get():
            prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
            copyTree(prevRunPath, netDataPath)
            if not self.keepTraining.get():
                nEpochs = 0
                
        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

          
        fnamesPos, weightsPos= self.__dataDict_toStrs(posTrainDict)
        fnamesNeg, weightsNeg= self.__dataDict_toStrs(negTrainDict)
        args= " -n %s --mode train -p %s -f %s --trueW %s --falseW %s"%(netDataPath, 
                      fnamesPos, fnamesNeg, weightsPos, weightsNeg)
        args+= " -e %s -l %s -r %s -m %s "%(nEpochs, self.learningRate.get(), self.l2RegStrength.get(),
                                          self.nModels.get())
        if not self.auto_stopping.get():
          args+=" -s"
          
        if not gpuToUse is None:
          args+= " -g %s"%(gpuToUse)
        if not numberOfThreads is None:
          args+= " -t %s"%(numberOfThreads)
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1)

    def predict(self, posTestDict, negTestDict, predictDict):
        """
            posTestDict, negTestDict, predictDict: { fnameToMetadata: { weight:int }
        """        
        netDataPath = self._getExtraPath('nnetData')
        if not os.path.isdir(netDataPath) and self.doContinue.get():
            prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
            copyTree(prevRunPath, netDataPath)

        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        outParticlesPath = self._getPath("particles.xmd")
        fnamesPred, weightsPred= self.__dataDict_toStrs(predictDict)
        
        args= " -n %s --mode score -i %s -o %s "%(netDataPath, fnamesPred, outParticlesPath)
                
        if posTestDict and negTestDict:
          fnamesPosTest, weightsPosTest= self.__dataDict_toStrs(posTestDict)
          fnamesNegTest, weightsNegTest= self.__dataDict_toStrs(negTestDict)
          args+= " --testingTrue %s --testingFalse %s "%(fnamesPosTest, fnamesNegTest)
          
        if not gpuToUse is None:
          args+= " -g %s"%(gpuToUse)
        if not numberOfThreads is None:
          args+= " -t %s"%(numberOfThreads)
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1)
        
    def createOutputStep(self):
        imgSet = self.predictSetOfParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet, updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self._getPath("particles.xmd"),
                                                       sortByLabel=md.MDL_ITEM_ID))
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)


    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        pass

    #--------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(md.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True

