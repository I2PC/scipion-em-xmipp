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


import os

from pyworkflow.utils.path import copyTree, makeFilePath
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtProcessParticles
import pyworkflow.em.metadata as md
from pyworkflow.em.packages.xmipp3.convert import writeSetOfParticles, setXmippAttributes

WRITE_TEST_SCORES=True
N_MAX_NEG_SETS= 5
class XmippProtScreenDeepLearning1(ProtProcessParticles):
    """ Protocol for screening particles using deep learning. """
    _label = 'screen deep learning'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('doContinue', params.BooleanParam, default=False,
                      label='use previously trained model?',
                      help='If you set to *Yes*, you should select a previous'
                      'run of type *%s* class and some of the input parameters'
                      'will be taken from it.' % self.getClassName())
        form.addParam('continueRun', params.PointerParam, pointerClass=self.getClassName(),
                      condition='doContinue', allowsNull=True,
                      label='Select previous run',
                      help='Select a previous run to continue from.')
        form.addParam('keepTraining', params.BooleanParam, default=True,condition='doContinue',
                      label='continue training on previously trainedModel?',
                      help='If you set to *Yes*, you should provide training set')

        form.addParam('inPosSetOfParticles', params.PointerParam, label="True particles",
                      pointerClass='SetOfParticles', allowsNull=True, condition="not doContinue or keepTraining",
                      help='Select a set of particles that are mostly true particles.\n'
                           'Recomended set: good Z-score of AND consensus of two pickers')

        form.addParam('numberOfNegativeSets', params.IntParam, default='1', condition="not doContinue or keepTraining",
                          label='Number of different negative dataset',
                          help='Data from all negative datasets will be used for training. Maximun number is 4.\n'
                       'Recomended negative data sets are 2:\n'
                   '1) Random picked (pick noise protocol)\n'
                   '2) Bad Z-score from multiple picker OR consensus\n')
        for num in range( 1, N_MAX_NEG_SETS):
            form.addParam('negativeSet_%d'%num, params.PointerParam, 
                            condition='(numberOfNegativeSets<=0 or numberOfNegativeSets >=%d) and (not doContinue or keepTraining)'%num,
                            label="Set of negative train particles %d"%num,
                            pointerClass='SetOfParticles', allowsNull=True,
                            help='Select the set of negative particles for training.')
            form.addParam('inNegWeight_%d'%num, params.IntParam, default='1',
                            condition='(numberOfNegativeSets<=0 or numberOfNegativeSets >=%d) and (not doContinue or keepTraining)'%num,
                            label="Weight of negative train particles %d"%num, allowsNull=True,
                            help='Select the weigth for the negative set of particles. The weight value '
                                 'indicates the number of times each image may be included at most per epoch. Positive '
                                 'particles are weighted with 1. If weight is -1, weight will be calculated such that '
                                 'the contribution of additional data is equal to the contribution of positive particles')

        form.addParam('predictSetOfParticles', params.PointerParam, label="Set of putative particles to predict",
                      pointerClass='SetOfParticles',
                      help='Select the set of putative particles particles to classify.')

        use_cuda=True
        if 'CUDA' in os.environ and not os.environ['CUDA']=="False":

            form.addParam('gpuToUse', params.IntParam, default='0',
                          label='Which GPU to use:',
                          help='Currently just one GPU will be use, by '
                               'default GPU number 0 You can override the default '
                               'allocation by providing other GPU number, p.e. 2'
                               '\nif -1 no gpu but all cpu cores will be used')
        else:
            use_cuda=False

        form.addParam('auto_stopping',params.BooleanParam, default=True, condition="not doContinue or keepTraining",
                      label='Auto stop training when convergency is detected?', expertLevel=params.LEVEL_ADVANCED,
                      help='If you set to *Yes*, the program will automatically stop training'
                      'if there is no improvement for consecutive 2 epochs, learning rate will be decreased'
                      'by a factor 10. If learningRate_t < 0.01*learningrate_0 training will stop. Warning: '
                      'Sometimes convergency seems to be reached, but after time, improvement can still happen. '
                      'Not recommended for very small data sets (<100 true particles)')
        
        form.addParam('nEpochs', params.FloatParam, label="Number of epochs", default=5.0, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue or keepTraining", help='Number of epochs for neural network training.')
        form.addParam('learningRate', params.FloatParam, label="Learning rate", default=1e-4, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue or keepTraining", help='Learning rate for neural network training')
        form.addParam('l2RegStrength', params.FloatParam, label="Regularization strength", default=1e-5, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue or keepTraining", help='L2 regularization for neural network weights. Make it '
                                                                  'bigger if suffering overfitting. Typical values range from 1e-1 to 1e-6')        
        form.addParam('nModels', params.IntParam, label="Number of models for ensemble", default=2, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue",
                      help='Number of models to fit in order to build an ensamble. Tipical values are 1 to 5. The more the better'
                      'until a point where no gain is obtained. Each model increases running time linearly')

        form.addSection(label='testingData')
        form.addParam('doTesting', params.BooleanParam, default=False,
                      label='Perform testing after training?',
                      help='If you set to *Yes*, you should select a testing positive set '
                      'and a testing negative set')
        form.addParam('testPosSetOfParticles', params.PointerParam, condition='doTesting',
                      label="Set of positive test particles",
                      pointerClass='SetOfParticles',
                      help='Select the set of ground true positive particles.')
        form.addParam('testNegSetOfParticles', params.PointerParam, condition='doTesting',
                      label="Set of negative test particles",
                      pointerClass='SetOfParticles',
                      help='Select the set of ground false positive particles.')

        if not use_cuda: form.addParallelSection(threads=8, mpi=0)

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
            dictONameToWeight= { fname: fnameToSetAndWeight[fname][-1] for fname in fnameToSetAndWeight if 
                                                        not fnameToSetAndWeight[fname][0] is None}
            if len(dictONameToWeight)==0:
                return None
            else:
                return dictONameToWeight

        posSetTrainDict={self._getExtraPath("inputTrueParticlesSet.xmd"):(self.inPosSetOfParticles.get(), 1)}
        negSetTrainDict={}
        for num in range( 1, N_MAX_NEG_SETS):
            if self.numberOfNegativeSets<=0 or self.numberOfNegativeSets >=num:
                negativeSetParticles= self.__dict__["negativeSet_%d"%num].get()
                negSetTrainDict[self._getExtraPath("negativeSet_%d.xmd"%num)]=(negativeSetParticles,
                                                                               self.__dict__["inNegWeight_%d"%num].get())
        setPredictDict={self._getExtraPath("predictSetOfParticles.xmd"): (self.predictSetOfParticles.get(),1) }

        setTestPosDict={self._getExtraPath("testTrueParticlesSet.xmd"):  (self.testPosSetOfParticles.get(),1)}
        setTestNegDict={self._getExtraPath("testFalseParticlesSet.xmd"): (self.testNegSetOfParticles.get(),1)}


        self._insertFunctionStep('convertInputStep', posSetTrainDict, negSetTrainDict, setPredictDict, 
                                                     setTestPosDict, setTestNegDict)
        self._insertFunctionStep('train', _getFname2WeightDict(posSetTrainDict), _getFname2WeightDict(negSetTrainDict))
        self._insertFunctionStep('predict', _getFname2WeightDict(setTestPosDict), _getFname2WeightDict(setTestNegDict), 
                                            _getFname2WeightDict(setPredictDict))
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self, *dataDicts):
        if (not self.doContinue.get()  or self.keepTraining.get() ) and self.nEpochs.get()>0:
            assert not self.inPosSetOfParticles.get() is None, "Positive particles must be provided for training if nEpochs!=0"
        for num in range( 1, N_MAX_NEG_SETS):
            if self.numberOfNegativeSets<=0 or self.numberOfNegativeSets >=num:
                negativeSetParticles= self.__dict__["negativeSet_%d"%num].get()
                if (not self.doContinue.get()  or self.keepTraining.get() ) and self.nEpochs.get()>0:
                    assert not negativeSetParticles is None, "Negative particles must be provided for training if nEpochs!=0"
        for dataDict in dataDicts:
          for fnameParticles in sorted(dataDict):
            print(fnameParticles, dataDict[fnameParticles][0], not dataDict[fnameParticles][0] is None)
            if not dataDict[fnameParticles][0] is None:
                writeSetOfParticles(dataDict[fnameParticles][0], fnameParticles)

    def train(self, posTrainDict, negTrainDict):
        """
        posTrainDict, negTrainDict: { fnameToMetadata: { weight:int }
        """
        netDataPath= self._getExtraPath('nnetData')

        if hasattr(self, 'gpuToUse'):
            numberOfThreads=None
            gpuToUse= self.gpuToUse.get()
        else:
            numberOfThreads=self.numberOfThreads.get()
            gpuToUse= None

        nEpochs= self.nEpochs.get()
        if self.doContinue.get():
            prevRunPath= self.continueRun.get()._getExtraPath('nnetData')
            if not self.keepTraining.get():
                nEpochs=0
        else:
          prevRunPath= None

        trainWorker(netDataPath, posTrainDict, negTrainDict, nEpochs, self.learningRate.get(), 
                    self.l2RegStrength.get(), self.auto_stopping.get(), 
                    self.nModels.get(), gpuToUse, numberOfThreads, prevRunPath= prevRunPath)

    def predict(self, posTestDict, negTestDict, predictDict):
        """
        posTestDict, negTestDict, predictDictt: { fnameToMetadata: { weight:int }
        """        
        netDataPath= self._getExtraPath('nnetData')

        if hasattr(self, 'gpuToUse'):        
            numberOfThreads=None
            gpuToUse= self.gpuToUse.get()
        else:
            numberOfThreads=self.numberOfThreads.get()
            gpuToUse= None

        outParticlesPath= self._getPath("particles.xmd")
        predictWorker( netDataPath, posTestDict, negTestDict, predictDict, outParticlesPath, 
                        gpuToUse, numberOfThreads)


    def createOutputStep(self):
        imgSet = self.predictSetOfParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet,
                            updateItemCallback=self._updateParticle,
                            itemDataIterator=md.iterRows(self._getPath("particles.xmd"), sortByLabel=md.MDL_ITEM_ID))
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



def writeNetShape(netDataPath, shape, nTrue, nModels):
  '''
  netDataPath= self._getExtraPath("nnetData")
  '''
  netInfoFname= os.path.join(netDataPath, "nnetInfo.txt")

  makeFilePath( netInfoFname )
  with open( netInfoFname, "w" ) as f:
      f.write("inputShape: %d %d %d\ninputNTrue: %d\nnModels: %d"%( shape+(nTrue, nModels)))

def loadNetShape(netDataPath):
  '''
  netDataPath= self._getExtraPath("nnetData")
  '''
  netInfoFname= os.path.join(netDataPath, "nnetInfo.txt")
  with open( netInfoFname ) as f:
      lines= f.readlines()
      dataShape= tuple( [ int(elem) for elem in lines[0].split()[1:] ] )
      nTrue= int(lines[1].split()[1])
      nModels= int(lines[2].split()[1])
  return dataShape, nTrue, nModels
  
def trainWorker(netDataPath, posTrainDict, negTrainDict, nEpochs, learningRate, l2RegStrength, auto_stop, 
                numModels, gpuToUse, numberOfThreads, prevRunPath=None):
    '''
    netDataPath=self._getExtraPath("nnetData")
    posTrainDict, negTrainDict: { fnameToMetadata:  weight:int } 
              e.g. : {'Runs/006003_XmippProtScreenDeepLearning1/extra/negativeSet_1.xmd': 1}
    learningRate: float
    prevRunPath: a path to previous run or None. Generally prevRun._getExtraPath('nnetData')
    '''
#    print(prevRunPath, netDataPath, nEpochs)
    import sys
    sys.stdout.flush()
    if prevRunPath:
          copyTree(prevRunPath, netDataPath)
          dataShape, nTrue, numModels= loadNetShape(netDataPath)
    else:
        if nEpochs==0:
            raise ValueError("Number of epochs >0 if not continuing from trained model")
 
    if nEpochs==0:
        print("training is not required")
        return
    from .deepConsensus_deepLearning1 import updateEnviron
    if gpuToUse>=0:        
        numberOfThreads=None
    else:
        gpuToUse= None        
        if numberOfThreads<0:
            import multiprocessing          
            numberOfThreads= multiprocessing.cpu_count()
    updateEnviron(gpuToUse)
                
    try:
        from .deepConsensus_deepLearning1 import DeepTFSupervised, DataManager, tf_intarnalError
    except ImportError:
        raise ValueError('''
     Error, tensorflow is not installed. Install it with:\n  ./scipion install tensorflow
     If gpu version of tensorflow desired, install cuda 8 and cudnn 6 or cuda 9 and cudnn 7 add to scipion.conf
     CUDA = True
     CUDA_VERSION = 8.0  or 9.0
     CUDA_HOME = /path/to/cuda-8
     CUDA_BIN = %(CUDA_HOME)s/bin
     CUDA_LIB = %(CUDA_HOME)s/lib64
     CUDNN_VERSION = 6 or 7
''')

    trainDataManager= DataManager( posSetDict= posTrainDict, negSetDict= negTrainDict)
    if prevRunPath:
        assert dataShape== trainDataManager.shape, "Error, data shape mismatch in input data compared to previous model"

    writeNetShape(netDataPath, trainDataManager.shape, trainDataManager.nTrue, numModels)
    assert numModels>=1, "Error, nModels<1"
    try:
        nnet = DeepTFSupervised(numberOfThreads= numberOfThreads, rootPath= netDataPath,
                                  numberOfModels=numModels)
        nnet.trainNet(nEpochs, trainDataManager, learningRate, l2RegStrength, auto_stop)
    except tf_intarnalError as e:
        if e._error_code==13:
            raise Exception("Out of gpu Memory. gpu # %d"%(gpuToUse))
        else:
            raise e
    del nnet

def predictWorker( netDataPath, posTestDict, negTestDict, predictDict, outParticlesPath,
                   gpuToUse, numberOfThreads):
    '''
    outParticlesPath= self._getPath("particles.xmd")
    posTestDict, negTestDict, predictDict: { fnameToMetadata:  weight:int }
    '''

    from .deepConsensus_deepLearning1 import  updateEnviron
    if gpuToUse>=0:        
        numberOfThreads=None
    else:
        gpuToUse= None        
        if numberOfThreads<0:
            import multiprocessing          
            numberOfThreads= multiprocessing.cpu_count()
    updateEnviron(gpuToUse)

    try:
        from .deepConsensus_deepLearning1 import DeepTFSupervised, DataManager, tf_intarnalError
    except ImportError:
        raise ValueError('''
     Error, tensorflow is not installed. Install it with:\n  ./scipion install tensorflow
     If gpu version of tensorflow desired, install cuda 8 and cudnn 6 or cuda 9 and cudnn 7 add to scipion.conf
     CUDA = True
     CUDA_VERSION = 8.0  or 9.0
     CUDA_HOME = /path/to/cuda-8
     CUDA_BIN = %(CUDA_HOME)s/bin
     CUDA_LIB = %(CUDA_HOME)s/lib64
     CUDNN_VERSION = 6 or 7
''')

    predictDataManager= DataManager( posSetDict= predictDict, negSetDict= None)
    dataShape, nTrue, numModels= loadNetShape(netDataPath)
    try:
        nnet = DeepTFSupervised(numberOfThreads= numberOfThreads, rootPath=netDataPath, 
                                numberOfModels=numModels)
    except tf_intarnalError as e:
        if e._error_code==13:
            raise Exception("Out of gpu Memory. gpu # %d"%(gpuToUse))
        else:
            raise e

    y_pred, label_Id_dataSetNumIterator= nnet.predictNet(predictDataManager)

    metadataPosList, metadataNegList= predictDataManager.getMetadata(None)
    for score, (isPositive, mdId, dataSetNumber) in zip(y_pred, label_Id_dataSetNumIterator):
        if isPositive==True:
            metadataPosList[dataSetNumber].setValue(md.MDL_ZSCORE_DEEPLEARNING1, score, mdId)
        else:
            metadataNegList[dataSetNumber].setValue(md.MDL_ZSCORE_DEEPLEARNING1, score, mdId)

    assert len(metadataPosList)==1, "Error, predict setOfParticles must contain one single object"
    metadataPosList[0].write(outParticlesPath)

    if posTestDict and negTestDict:
      testDataManager= DataManager(posSetDict= posTestDict,
                           negSetDict= negTestDict, validationFraction=0)
      print("Evaluating test set")
      global_auc, global_acc, y_labels, y_pred_all= nnet.evaluateNet(testDataManager)
      if WRITE_TEST_SCORES:
        makeFilePath( os.path.join(netDataPath, "testPredictions.txt") )
        with open( os.path.join(netDataPath, "testPredictions.txt"), "w") as f:
          f.write("label score\n")
          for l, s in zip(y_labels, y_pred_all):
            f.write("%d %f\n"%(l, s))

