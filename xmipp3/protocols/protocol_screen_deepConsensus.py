# **************************************************************************
# *
# * Authors:    Ruben Sanchez Garcia (rsanchez@cnb.csic.es)
# *             David Maluenda (dmaluenda@cnb.csic.es)
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Deep Consensus picking protocol
"""
import os
from glob import glob
import six
import json

from pyworkflow import VERSION_2_0
from pyworkflow.utils.path import makePath, cleanPattern, cleanPath, copyTree, createLink
from pyworkflow.protocol.constants import *
from pyworkflow.em.constants import RELATION_CTF, ALIGN_NONE
from pyworkflow.em.convert import ImageHandler
from pyworkflow.em.data import SetOfCoordinates, Coordinate, SetOfParticles
from pyworkflow.em.protocol import ProtParticlePicking, ProtUserSubSet

import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.em.metadata as MD

import xmippLib as xmipp
import xmipp3
from xmipp3 import XmippProtocol
from xmipp3.protocols.protocol_pick_noise import pickNoise_prepareInput
from xmipp3.convert import readSetOfParticles, setXmippAttributes, micrographToCTFParam,\
                           writeSetOfParticles, writeSetOfCoordinates, readSetOfCoordsFromPosFnames
from xmipp3.utils import validateDLtoolkit

MIN_NUM_CONSENSUS_COORDS = 256

try:
  from xmippPyModules.deepConsensusWorkers.deepConsensus_networkDef import DEEP_PARTICLE_SIZE
except ImportError as e:
  DEEP_PARTICLE_SIZE = 128
  
class XmippProtScreenDeepConsensus(ProtParticlePicking, XmippProtocol):
    """ Protocol to compute a smart consensus between different particle picking
        algorithms. The protocol takes several Sets of Coordinates calculated
        by different programs and/or different parameter settings. Let's say:
        we consider N independent pickings. Then, a neural network is trained
        using different subset of picked and not picked cooridantes. Finally,
        a coordinate is considered to be a correct particle according to the
        neural network predictions.
    """
    _label = 'deep consensus picking'
    _lastUpdateVersion = VERSION_2_0

    CONSENSUS_COOR_PATH_TEMPLATE="consensus_coords_%s"
    CONSENSUS_PARTS_PATH_TEMPLATE="consensus_parts_%s"
    PRE_PROC_MICs_PATH="preProcMics"

    ADD_DATA_TRAIN_TYPES = ["None", "Precompiled", "Custom"]
    ADD_DATA_TRAIN_NONE = 0
    ADD_DATA_TRAIN_PRECOMP = 1
    ADD_DATA_TRAIN_CUST = 2

    ADD_MODEL_TRAIN_TYPES = ["New", "Pretrained", "PreviousRun"]
    ADD_MODEL_TRAIN_NEW = 0
    ADD_MODEL_TRAIN_PRETRAIN = 1
    ADD_MODEL_TRAIN_PREVRUN = 2

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_SERIAL

    def _defineParams(self, form):
        # GPU settings
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Use GPU (vs CPU)",
                       help="Set to true if you want to use GPU implementation ")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")
                            
        form.addParallelSection(threads=2, mpi=0)
        
        form.addSection(label='Input')
        
        form.addParam('modelInitialization', params.EnumParam,
                      choices=self.ADD_MODEL_TRAIN_TYPES,
                      default=self.ADD_MODEL_TRAIN_NEW,
                      label='Select model type',
                      help='If you set to *%s*, a new model randomly initialized will be '
                           'employed. If you set to *%s* a pretrained model will be used. '
                           'If you set to *%s*, a model trained in a previous run, within '
                           'this project, will be employed'
                           % tuple(self.ADD_MODEL_TRAIN_TYPES))
        #CONTINUE FROM PREVIOUS TRAIN
                   
        form.addParam('continueRun', params.PointerParam,
                      pointerClass=self.getClassName(),
                      condition='modelInitialization== %s'%self.ADD_MODEL_TRAIN_PREVRUN, allowsNull=True,
                      label='Select previous run',
                      help='Select a previous run to continue from.')
        form.addParam('skipTraining', params.BooleanParam,
                      default=False, condition='modelInitialization!= %s '%self.ADD_MODEL_TRAIN_NEW,
                      label='Skip training and score directly with pretrained model?',
                      help='If you set to *No*, you should provide training set. If set to *Yes* '
                            'the coordinates will be directly scored using the pretrained/previous model')


        form.addParam('inputCoordinates', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates', allowsNull=False,
                      label="Input coordinates",
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.FloatParam, default=0.1,
                      label="Relative Radius", expertLevel=params.LEVEL_ADVANCED,
                      validators=[params.Positive],
                      help="All coordinates within this radius "
                           "(as fraction of particle size) "
                           "are presumed to correspond to the same particle")

        form.addSection(label='Preprocess')
        form.addParam('notePreprocess', params.LabelParam,
                      label='How to extract particles from micrograph',
                      help='Our method, internally, uses particles that are '
                           'extracted from preprocess micrographs. '
                           'Preprocess steps are:\n'
                           '1) mic donwsampling to the required size such that '
                           'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                           ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                           '2) mic normalization to 0 mean and 1 std and mic contrast inversion to have white particles.\n '
                           '  E.g. '
                           ' xmipp_transform_normalize -i out1/101_movie_aligned.mrc -o out2/101_movie_aligned.mrc --method '
                           'OldXmipp [ --invert ]\n'
                           '3) particles extraction.\n   E.g. xmipp_micrograph_scissor  -i out2/101_movie_aligned.mrc '
                           '--pos particles@Runs/101_movie_aligned.pos -o out3/105_movie_aligned_particles '
                           ' --Xdim 128 --downsampling newSamplingRate --fillBorders  ( Correct your coordinates with '
                           'newSamplingRate if needed)\n'
                           '4) OPTIONAL: phase flipping using CTF.\n xmipp_ctf_phase_flip  -i '
                           'particles/105_movie_aligned_noDust.xmp -o particles/105_movie_aligned_flipped.xmp '
                           '--ctf ctfPath/105_movie_aligned.ctfParam --sampling newSamplingRate')

        form.addParam('skipInvert', params.BooleanParam, default=False,
                      label='Did you invert the micrographs contrast (particles are bright)?', 
                      help='If you invert the contrast, your particles will be white over a black background in the micrograph. '
                           'We use white particles. Select *No* if you already have inverted the constrast in the micrograph'
                           ' so that we can extract white particles directly')
        
        form.addParam('ignoreCTF', params.BooleanParam, default=True,
                      label='Ignore CTF',
                      help='Deep Consensus extracts particles. Do you want to ignore CTF for '
                            'particle extraction')
                           
        form.addParam('ctfRelations', params.RelationParam, allowsNull=True,
                      relationName=RELATION_CTF, condition="not ignoreCTF",
                      attributeName='_getInputMicrographs',
                      label='CTF estimation',
                      help='Choose some CTF estimation related to input '
                           'micrographs. \nCTF estimation is needed if you '
                           'want to do phase flipping or you want to '
                           'associate CTF information to the particles.')
        

        form.addSection(label='Training')
        
        form.addParam('nEpochs', params.FloatParam,
                      label="Number of epochs", default=5.0,
                      condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
                      help='Number of epochs for neural network training.')
        form.addParam('learningRate', params.FloatParam,
                      label="Learning rate", default=1e-4,
                      condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
                      help='Learning rate for neural network training')
        form.addParam('auto_stopping',params.BooleanParam,
                      label='Auto stop training when convergency is detected?',
                      default=True, condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
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
                      condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
                      help='L2 regularization for neural network weights.'
                           'Make it bigger if suffering overfitting (validation acc decreases but training acc increases)\n'
                           'Typical values range from 1e-1 to 1e-6')
        form.addParam('nModels', params.IntParam,
                      label="Number of models for ensemble",
                      default=3, expertLevel=params.LEVEL_ADVANCED,
                      condition="modelInitialization==%s or (not skipTraining and modelInitialization==%s)"%(
                                                          self.ADD_MODEL_TRAIN_NEW, self.ADD_MODEL_TRAIN_PRETRAIN),
                      help='Number of models to fit in order to build an ensamble. '
                           'Tipical values are 1 to 5. The more the better '
                           'until a point where no gain is obtained. '
                           'Each model increases running time linearly')
                           
        form.addParam('doTesting', params.BooleanParam, default=False,
                      label='Perform testing after training?', expertLevel=params.LEVEL_ADVANCED,
                      condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
                      help='If you set to *Yes*, you should select a testing '
                           'positive set and a testing negative set')
        form.addParam('testTrueSetOfParticles', params.PointerParam,
                      label="Set of positive test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles',condition='doTesting',
                      help='Select the set of ground true positive particles.')
        form.addParam('testFalseSetOfParticles', params.PointerParam,
                      label="Set of negative test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles', condition='doTesting',
                      help='Select the set of ground false positive particles.')

        form.addSection(label='Additional training data')
        form.addParam('addTrainingData', params.EnumParam,
                      condition="modelInitialization==%s or not skipTraining"%self.ADD_MODEL_TRAIN_NEW,
                      choices=self.ADD_DATA_TRAIN_TYPES,
                      default=self.ADD_DATA_TRAIN_PRECOMP,
                      label='Additional training data',
                      help='If you set to *%s*, you should select positive and/or '
                           'negative sets of particles. Regard that our method, '
                           'internally, uses particles that are extracted from '
                           'preprocess micrographs. Steps are:\n'
                           '1) mic donwsampling to the required size such that '
                           'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                           ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                           '2) mic normalization to 0 mean and 1 std and mic contrast inversion to have white particles.\n '
                           'E.g. '
                           ' xmipp_transform_normalize -i out1/101_movie_aligned.mrc -o out2/101_movie_aligned.mrc --method '
                           'OldXmipp [ --invert ]\n'
                           '3) particles extraction.\n   E.g. xmipp_micrograph_scissor  -i out2/101_movie_aligned.mrc '
                           '--pos particles@Runs/101_movie_aligned.pos -o out3/105_movie_aligned_particles '
                           ' --Xdim 128 --downsampling newSamplingRate --fillBorders  ( Correct your coordinates with '
                           'newSamplingRate if needed)\n'
                           '4) OPTIONAL: phase flipping using CTF.\n xmipp_ctf_phase_flip  -i '
                           'particles/105_movie_aligned_noDust.xmp -o particles/105_movie_aligned_flipped.xmp '
                           '--ctf ctfPath/105_movie_aligned.ctfParam --sampling newSamplingRate\n'
                           'Then, particles are extracted with no further alteration.\n'
                           'Please ensure that the additional particles have been '
                           'preprocessed as indicated before.\n\n'
                           'Alternatively, you can use a precompiled dataset '
                           'by selecting *%s* or simply, use only the '
                           'input coorditanes (*%s*).'
                           %tuple(self.ADD_DATA_TRAIN_TYPES) )
        form.addParam('trainTrueSetOfParticles', params.PointerParam,
                      label="Positive train particles 128px (optional)",
                      pointerClass='SetOfParticles', allowsNull=True,
                      condition=("(modelInitialization==%s or not skipTraining ) "+
                                "and addTrainingData==%s")%(self.ADD_MODEL_TRAIN_NEW, self.ADD_DATA_TRAIN_CUST),
                      help='Select a set of true positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainPosWeight', params.IntParam, default='1',
                      label="Weight of positive additional train particles",
                      condition=("(modelInitialization==%s or not skipTraining ) "+
                                "and addTrainingData==%s")%(self.ADD_MODEL_TRAIN_NEW, self.ADD_DATA_TRAIN_CUST),
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'positive particles.The weight value indicates '
                           'internal particles are weighted with 1. '
                           'If weight is -1, weight will be calculated such that '
                           'the contribution of additional data is equal to '
                           'the contribution of internal particles')
        form.addParam('trainFalseSetOfParticles', params.PointerParam,
                      label="Negative train particles 128px (optional)",
                      pointerClass='SetOfParticles',  allowsNull=True,
                      condition=("(modelInitialization==%s or not skipTraining ) "+
                                "and addTrainingData==%s")%(self.ADD_MODEL_TRAIN_NEW, self.ADD_DATA_TRAIN_CUST),
                      help='Select a set of false positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainNegWeight', params.IntParam, default='1',
                      label="Weight of negative additional train particles",
                      condition=("(modelInitialization==%s or not skipTraining ) "+
                                "and addTrainingData==%s")%(self.ADD_MODEL_TRAIN_NEW, self.ADD_DATA_TRAIN_CUST),
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'negative particles. The weight value indicates '
                           'the number of times each image may be included at '
                           'most per epoch. Deep consensus internal particles '
                           'are weighted with 1. If weight is -1, weight '
                           'will be calculated such that the contribution of '
                           'additional data is equal to the contribution of '
                           'internal particles')

    def _validate(self):
        errorMsg = []
        if self._getBoxSize()< DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, too small particles (needed 128 px), "
                          "have you provided already downsampled micrographs? "
                          "If so, use original ones")
        if not self.ignoreCTF.get() and self.ctfRelations.get() is None:
          errorMsg.append("Error, CTFs must be provided to compute phase flip. "
                          "Please, provide a set of CTFs.")

        if  self.trainTrueSetOfParticles.get() and self.trainTrueSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, trainTrueSetOfParticles needed to be 128 px")

        if  self.trainFalseSetOfParticles.get() and self.trainFalseSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, trainFalseSetOfParticles needed to be 128 px")

        if  self.testTrueSetOfParticles.get() and self.testTrueSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, testTrueSetOfParticles needed to be 128 px")
          
        if  self.testFalseSetOfParticles.get() and self.testFalseSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, testFalseSetOfParticles needed to be 128 px")
          
        if len(self.inputCoordinates)==1  and not self.justPredict():
          errorMsg.append("Error, just one coordinate set provided but trained desired. Select pretrained "+
                          "model or previous run model and *No* continue training from previous trained model "+
                          " to score coordiantes directly or add another set of particles and continue training")
        errorMsg = validateDLtoolkit(errorMsg, model="deepConsensus",
                                     assertModel=self.addTrainingData.get()==self.ADD_DATA_TRAIN_PRECOMP)
        return errorMsg

#--------------------------- INSERT steps functions ---------------------------

    def _doContinue(self):
        return self.modelInitialization.get()== self.ADD_MODEL_TRAIN_PREVRUN
        
    def justPredict(self):
      return self.skipTraining.get()==True
      
    def _usePretrainedModel(self):
        return self.modelInitialization.get()== self.ADD_MODEL_TRAIN_PRETRAIN
        
    def _insertAllSteps(self):

        self.inputMicrographs = None
        self.boxSize = None
        self.coordinatesDict = {}
        deps = []
        deps.append(self._insertFunctionStep("initializeStep", prerequisites=deps))
        depsPrepMic = self._insertFunctionStep("preprocessMicsStep", prerequisites=deps)
        depsPrepMic=[depsPrepMic]
        # OR before noise always, as noise needs to need the number of OR coords
        depsOr_cons = self.insertCaculateConsensusSteps('OR', prerequisites=depsPrepMic)
        depsOr = self.insertExtractPartSteps('OR', prerequisites=depsOr_cons)
        depsTrain = depsOr

        if len(self.inputCoordinates)>1 and  not self.skipTraining.get():
          depNoise = self._insertFunctionStep('pickNoise', prerequisites=depsOr)
          depsNoise = self.insertExtractPartSteps('NOISE', prerequisites=[depNoise])

          depsAnd = self.insertCaculateConsensusSteps('AND', prerequisites=depsOr)
          depsAnd = self.insertExtractPartSteps('AND', prerequisites=depsAnd)

          depsTrain = depsTrain + depsNoise + depsAnd
          depTrain = self._insertFunctionStep('trainCNN', prerequisites=depsTrain)
        else:
          depTrain=None
        depPredict = self._insertFunctionStep('predictCNN', prerequisites= [depTrain] if not depTrain is None else [])
        self._insertFunctionStep('createOutputStep', prerequisites=[depPredict])

    def initializeStep(self):
        """
            Create paths where data will be saved
        """

        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get():
            writeSetOfParticles(self.testTrueSetOfParticles.get(),
                                self._getExtraPath("testTrueParticlesSet.xmd"))
            writeSetOfParticles(self.testFalseSetOfParticles.get(),
                                self._getExtraPath("testFalseParticlesSet.xmd"))

        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_CUST:
            if self.trainTrueSetOfParticles.get():
                writeSetOfParticles(self.trainTrueSetOfParticles.get(),
                                    self._getExtraPath("trainTrueParticlesSet.xmd"))
            if self.trainFalseSetOfParticles.get():
                writeSetOfParticles(self.trainFalseSetOfParticles.get(),
                                    self._getExtraPath("trainFalseParticlesSet.xmd"))
        elif self.addTrainingData.get() == self.ADD_DATA_TRAIN_PRECOMP:
            writeSetOfParticles(self.retrieveTrainSets(),
                                self._getTmpPath("addNegTrainParticles.xmd"))

        if self.checkIfPrevRunIsCompatible(""):
            for mode in ["OR", "AND", "NOISE"]:
              print("copying particles %s"%(mode))
              newDataPath= "particles_%s.xmd" % mode
              createLink( self.continueRun.get()._getExtraPath(newDataPath), 
                        self._getExtraPath(newDataPath))
              newDataPath= "parts_%s" % mode
              createLink( self.continueRun.get()._getExtraPath(newDataPath), 
                        self._getExtraPath(newDataPath))
            print("copying OR coordinates")
            newDataPath= (self.CONSENSUS_COOR_PATH_TEMPLATE % "OR")
            createLink( self.continueRun.get()._getExtraPath(newDataPath), 
                        self._getExtraPath(newDataPath))
        else:
          if self.checkIfPrevRunIsCompatible( "mics_"):
            sourcePath= self.continueRun.get()._getTmpPath(self.PRE_PROC_MICs_PATH )
            print("copying mics from %s"%(sourcePath))
            createLink( sourcePath, self._getTmpPath(self.PRE_PROC_MICs_PATH ))
          else:
            makePath(self._getTmpPath(self.PRE_PROC_MICs_PATH))

          if self.checkIfPrevRunIsCompatible( "coords_"):
            for mode in ["OR", "AND", "NOISE"]:
              print("copying coordinates %s"%(mode))
              newDataPath= (self.CONSENSUS_COOR_PATH_TEMPLATE % mode)
              createLink( self.continueRun.get()._getExtraPath(newDataPath), 
                        self._getExtraPath(newDataPath))
          else:
              for mode in ["AND", "OR", "NOISE"]:
                consensusCoordsPath = self.CONSENSUS_COOR_PATH_TEMPLATE % mode
                makePath(self._getExtraPath(consensusCoordsPath))
         
        preprocessParamsFname= self._getExtraPath("preprocess_params.json")
        preprocParams= self.getPreProcParamsFromForm()
        with open(preprocessParamsFname, "w") as f:
          json.dump(preprocParams, f)
      
    def retrieveTrainSets(self):
        """ Retrieve, link and return a setOfParticles
            corresponding to the NegativeTrain DeepConsensus trainning set
            with certain extraction conditions (phaseFlip/invContrast)
        """
        prefixYES = ''
        prefixNO = 'no'
        #We always work with inverted contrast particles
        modelType = "negativeTrain_%sPhaseFlip_Invert.mrcs" % (
                    prefixNO if self.ignoreCTF.get() else prefixYES) # mics will be always internally inverted if not done before
        modelPath = xmipp3.Plugin.getModel("deepConsensus", modelType)
        modelFn = self._getTmpPath(modelType)
        pwutils.createLink(modelPath, modelFn)

        tmpSqliteSuff = "AddTrain"
        partSet = self._createSetOfParticles(tmpSqliteSuff)
        img = SetOfParticles.ITEM_TYPE()

        imgh = ImageHandler()
        _, _, _, n = imgh.getDimensions(modelFn)
        if n > 1:
            for index in range(1, n + 1):
                img.cleanObjId()
                img.setMicId(9999)
                img.setFileName(modelFn)
                img.setIndex(index)
                partSet.append(img)
        partSet.setAlignment(ALIGN_NONE)

        cleanPath(self._getPath("particles%s.sqlite"%tmpSqliteSuff))
        return partSet

    def _getInputMicrographs(self):
        if not hasattr(self, "inputMicrographs") or not self.inputMicrographs:
            if len(self.inputCoordinates)==0:
              print("WARNING. PROVIDE MICROGRAPHS FIRST")
            else:
              inputMicrographs = self.inputCoordinates[0].get().getMicrographs()
              if inputMicrographs is None:
                raise ValueError("there are problems with your coordiantes, they do not have associated micrographs ")
              self.inputMicrographs= inputMicrographs
        return self.inputMicrographs

    def _getBoxSize(self):
        if not hasattr(self, "boxSize") or not self.boxSize:
            firstCoords = self.inputCoordinates[0].get()
            self.boxSize = firstCoords.getBoxSize()
            self.downFactor = self.boxSize / float(DEEP_PARTICLE_SIZE)
        return self.boxSize

    def _getDownFactor(self):

        if not hasattr(self, "downFactor") or not self.downFactor:
          firstCoords = self._getInputMicrographs()
          self.boxSize= firstCoords.getBoxSize()
          self.downFactor = self.boxSize /float(DEEP_PARTICLE_SIZE)
          assert self.downFactor >= 1, \
              "Error, the particle box size must be greater or equal than 128."

        return self.downFactor


    def insertCaculateConsensusSteps(self, mode, prerequisites):
        
        deps = []
        outCoordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE% mode)
        if not self.checkIfPrevRunIsCompatible( "coords_"):
          dep_ = prerequisites
          makePath(outCoordsDataPath)
          newDep=self._insertFunctionStep('calculateCoorConsensusStep', outCoordsDataPath, mode, prerequisites=prerequisites)
          deps = [newDep]
          newDep = self._insertFunctionStep('loadCoords', outCoordsDataPath,
                                          mode, prerequisites=deps)
          deps = [newDep]
        else:
          deps= prerequisites
        return deps
        
    def calculateCoorConsensusStep(self, outCoordsDataPath, mode):
      Tm = []
      for coordinatesP in self.inputCoordinates:
          mics= coordinatesP.get().getMicrographs()
          Tm.append(mics.getSamplingRate())
      nCoordsSets= len(Tm)
      inputCoordsFnames={}
      for coord_num, coordinatesP in enumerate(self.inputCoordinates):
          tmpPosDir= self._getTmpPath("input_coords_%d"%(coord_num))
          makePath(tmpPosDir)
          writeSetOfCoordinates(tmpPosDir, coordinatesP.get(), scale=float(Tm[coord_num])/float(Tm[0]))
          for posFname in os.listdir(tmpPosDir):
              baseName, extension= os.path.basename(posFname).split(".")
              if extension=="pos":
                if baseName not in inputCoordsFnames:
                    inputCoordsFnames[baseName]=["None"]*nCoordsSets
                inputCoordsFnames[baseName][coord_num]= os.path.join(tmpPosDir, posFname)
      inputFileStr="#pos_i"
      for baseName in inputCoordsFnames:
         fnames= inputCoordsFnames[baseName]
         inputFileStr+=" ".join(fnames)+"\n"
             
      assert len(inputFileStr)>0, "Error, no consensus can be computed as there are mismatch in coordinate sets filenames"
      consensus = -1 if mode=="AND" else 1
      configFname= self._getTmpPath("consensus_%s_inputs.txt"%(mode) )
      with open(configFname, "w") as f:
          f.write(inputFileStr)
          
      args="-i %s -s %d -c %d -d %f -o %s -t %d"%(configFname, self._getBoxSize(), consensus, self.consensusRadius.get(),
                                                 outCoordsDataPath, self.numberOfThreads.get())
      self.runJob('xmipp_coordinates_consensus', args, numberOfMpi=1)

        
    def loadCoords(self, posCoorsPath, mode):
        boxSize = self._getBoxSize()
        inputMics = self._getInputMicrographs()
        sqliteName= self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode)+".sqlite"
        if os.path.isfile(self._getExtraPath(sqliteName)):
            cleanPath(self._getExtraPath(sqliteName))
        setOfCoordinates= readSetOfCoordsFromPosFnames(posCoorsPath, setOfInputCoords= self.inputCoordinates[0].get(),
                                   sqliteOutName= sqliteName, write=True )
        print("Coordinates %s size: %d"%( mode, setOfCoordinates.getSize()) )
        assert setOfCoordinates.getSize() > MIN_NUM_CONSENSUS_COORDS, \
                ("Error, the consensus (%s) of your input coordinates was too small (%s). "+
                "It must be > %s. Try a different input..."
                )% (mode, str(setOfCoordinates.getSize()), str(MIN_NUM_CONSENSUS_COORDS))

        self.coordinatesDict[mode]= setOfCoordinates
        
    def pickNoise(self):
        if self.checkIfPrevRunIsCompatible("coords_"):
          print("using previous round noise particles")
        else:
          outputPosDir= self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % "NOISE")
          if not "OR" in self.coordinatesDict:  # fill self.coordinatesDict['OR']
              self.loadCoords( self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR'),
                               'OR')

          argsDict=pickNoise_prepareInput(self.coordinatesDict['OR'], self._getTmpPath())
          argsDict["outputPosDir"]= outputPosDir
          argsDict["nThrs"] = self.numberOfThreads.get()
          argsDict["nToPick"]=-1
          args=(" -i %(mics_dir)s -c %(inCoordsPosDir)s -o %(outputPosDir)s -s %(boxSize)s "+
                "-n %(nToPick)s -t %(nThrs)s")%argsDict
          
          if not self.checkIfPrevRunIsCompatible( "coords_"):                
              self.runJob('xmipp_pick_noise', args, numberOfMpi=1)
              
          self.loadCoords(outputPosDir, 'NOISE')
          
        
    def getPreProcParamsFromForm(self):
        mics_= self._getInputMicrographs()
        mic = mics_.getFirstItem()
        fnMic = mic.getFileName()
        pathToMics= os.path.split(fnMic)[0]
        pathToCtfs= "None"
        
        if not self.ignoreCTF.get():
          pathToCtfs=  os.path.split(self.ctfRelations.get().getFileName())[0]

        paramsInfo={"mics_ignoreCTF":self.ignoreCTF.get(), "mics_skipInvert":self.skipInvert.get(), 
                    "mics_pathToMics":pathToMics, "mics_pathToCtfs": pathToCtfs}
        coordsNames=[]            
        for inputCoords in self.inputCoordinates:
          coordsNames.append( inputCoords.get().getFileName() )
        coordsNames= tuple(sorted(coordsNames))
        paramsInfo["coords_pathCoords"]= coordsNames

        return paramsInfo

    def checkIfPrevRunIsCompatible(self, inputType=""):
        '''
        inputType can be mics_ or coords_ or ""
        '''
        def _makeTupleIfList(candidateList):
          if isinstance(candidateList, list):
            return tuple(candidateList)
          elif isinstance(candidateList, six.string_types):
            return tuple([candidateList])
          else:
            return candidateList
        try:
          preprocParams= self.getPreProcParamsFromForm()
        except Exception:
          return False        
        preprocParams= { k:_makeTupleIfList(preprocParams[k]) for k in preprocParams if k.startswith(inputType) }
        if self._doContinue():
            preprocessParamsFname = self.continueRun.get()._getExtraPath("preprocess_params.json")
            with open(preprocessParamsFname) as f:
              preprocParams_loaded = json.load(f)

            preprocParams_loaded= { k:_makeTupleIfList(preprocParams_loaded[k]) for k in preprocParams_loaded 
                                            if k.startswith(inputType) }
           
            for key in preprocParams_loaded:
              if "path" in key:
                for val in preprocParams_loaded[key]:
                  if not val=="None" and not os.path.exists(val):
                    return False

        
            shared_items = {k: preprocParams_loaded[k] for k in preprocParams_loaded if 
                                  k in preprocParams and preprocParams_loaded[k] == preprocParams[k]}
  
                                    
            return len(shared_items)==len(preprocParams) and len(shared_items)==len(preprocParams_loaded) and \
                   len(shared_items)>0

        return False
        
    def getMicsIds(self, filterOutNoCoords=False):
      
        if not filterOutNoCoords:
          return self._getInputMicrographs().getIdSet()
        micIds= set([])
        micFnames= set([])
        for coordinatesP in self.inputCoordinates:
            for coord in coordinatesP.get():
              micIds.add( coord.getMicId())
              micFnames.add( coord.getMicName() )
        return sorted( micIds )
      
    def preprocessMicsStep(self):
        if not self.checkIfPrevRunIsCompatible( "mics_"): #or not os.path.isdir(self._getTmpPath(self.PRE_PROC_MICs_PATH)):
            micIds = self.getMicsIds(filterOutNoCoords=True)
            samplingRate = self._getInputMicrographs().getSamplingRate()
            mics_ = self._getInputMicrographs()
            micsFnameSet = { mics_[micId].getMicName(): mics_[micId].getFileName() for micId in micIds }
            if self.ignoreCTF.get():
              preproMicsContent="#mics\n"
              for micName in micsFnameSet:
                preproMicsContent+= "%s\n"%(micsFnameSet[micName])
            else:
              preproMicsContent="#mics ctfs\n"
              setOfMicCtf= self.ctfRelations.get()
              if setOfMicCtf.getSize() != len(micsFnameSet):
                raise ValueError("Error, there are different number of CTFs compared to "+
                                  "the number of micrographs where particles were picked")
              else:
                assert setOfMicCtf is not None, "Error, CTFs must be provided to compute phase flip"
                for ctf in setOfMicCtf:
                  ctf_mic = ctf.getMicrograph()
                  ctfMicName = ctf_mic.getMicName()
                  if ctfMicName in micsFnameSet:
                    ctf_mic.setCTF(ctf)
                    ctfMicName= micsFnameSet[ctfMicName]
                    fnCTF = self._getTmpPath("%s.ctfParam" % os.path.basename(ctfMicName))
                    micrographToCTFParam(ctf_mic, fnCTF) 
                    preproMicsContent+= "%s %s\n"%(ctfMicName, fnCTF)
                          

            inputsFname= self._getTmpPath("preprocMic_inputs.txt")
            ouputDir= self._getTmpPath(self.PRE_PROC_MICs_PATH)
            nThrs= self.numberOfThreads.get()
            with open(inputsFname, "w") as f:
              f.write(preproMicsContent)
            downFactor = self._getDownFactor()
            args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, samplingRate, downFactor, ouputDir, nThrs)
            if not self.skipInvert.get():
              args+=" --invert_contrast"
                
            if not self.ignoreCTF.get():
              args+=" --phase_flip"
              
            self.runJob('xmipp_preprocess_mics', args, numberOfMpi=1)
        
    def insertExtractPartSteps(self, mode, prerequisites):
        deps = []
        if not self.checkIfPrevRunIsCompatible(""):
          newDep = self._insertFunctionStep("extractParticles", mode, prerequisites= prerequisites)
          deps.append(newDep)
          newDep= self._insertFunctionStep("joinSetOfParticlesStep", mode, prerequisites= deps)
          deps.append(newDep)
        else:
          deps = prerequisites
          
        return deps

    def extractParticles(self, mode):

        downFactor = self._getDownFactor()
        mics_ = self._getInputMicrographs()
        micsFnameSet = {}
        preprocMicsPath= self._getTmpPath(self.PRE_PROC_MICs_PATH)
        for micFname in os.listdir(preprocMicsPath):
          micFnameBase= micFname.split(".")[0]
          micFname= os.path.join(preprocMicsPath, micFname)
          micsFnameSet[micFnameBase]= micFname
        extractCoordsContent="#mics coords\n"
        posDir= self._getExtraPath( self.CONSENSUS_COOR_PATH_TEMPLATE%mode )
        for posFname in os.listdir(posDir):
          posNameBase=  posFname.split(".")[0]
          posFname= os.path.join(posDir, posFname)
          if posNameBase in micsFnameSet:
            extractCoordsContent+= "%s particles@%s\n"%(micsFnameSet[posNameBase], posFname)
          else:
            print("WARNING, no micFn for coords %s"%(posFname))
          
        inputsFname= self._getTmpPath("extractParticles_inputs.txt")
        ouputDir= self._getExtraPath(self.CONSENSUS_PARTS_PATH_TEMPLATE% mode)
        makePath(ouputDir)
        nThrs= self.numberOfThreads.get()
        with open(inputsFname, "w") as f:
          f.write(extractCoordsContent)
        downFactor= self._getDownFactor()
        args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, DEEP_PARTICLE_SIZE, downFactor, ouputDir, nThrs)

        self.runJob('xmipp_extract_particles', args, numberOfMpi=1)   
        
    def joinSetOfParticlesStep( self, mode):
        #Create images.xmd metadata joining from different .stk
        fnImages = self._getExtraPath("particles_%s.xmd" % mode)
        imgsXmd = MD.MetaData()
        posFiles = glob(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode, '*.pos'))

        for posFn in posFiles:
          xmdFn = self._getExtraPath(self.CONSENSUS_PARTS_PATH_TEMPLATE%mode,
                                     pwutils.replaceBaseExt(posFn, "xmd"))
          if os.path.exists(xmdFn):
            mdFn = MD.MetaData(xmdFn)
            mdPos = MD.MetaData('particles@%s' % posFn)
            mdPos.merge(mdFn) 
            imgsXmd.unionAll(mdPos)
          else:
            self.warning("The coord file %s wasn't used for extraction! "
                         % os.path.basename(posFn))
            self.warning("Maybe you are extracting over a subset of micrographs")
        imgsXmd.write(fnImages)
        
    def __dataDict_toStrs(self, dataDict):
        fnamesStr=[]
        weightsStr=[]
        for fname in dataDict:
          fnamesStr.append(fname)
          weightsStr.append(str(dataDict[fname]) )
        return ":".join(fnamesStr), ":".join(weightsStr)
        
    def _getEffectiveNumPartsTrain(self, dictTrueData):
        nParts=0
        for mdPath in dictTrueData:
          mdObject = MD.MetaData(mdPath)
          nParts+= mdObject.size()
        return nParts

    def __retrievePreTrainedModel(self, netDataPath, effectiveSize=-1):
        if effectiveSize==-1:
          effectiveSize=int(5e4)
        modelTypeDir= "keras_models/%sPhaseFlip_Invert/nnetData_%d/tfchkpoints_0" % (
                            "no" if self.ignoreCTF.get() else "", effectiveSize)
        modelTypeDir= xmipp3.Plugin.getModel("deepConsensus", modelTypeDir)
        
        for i in range(self.nModels.get()):
          targetPath= os.path.join(netDataPath, "tfchkpoints_%d"%(i))
          print(targetPath, modelTypeDir)
          copyTree(modelTypeDir, targetPath)
          
    def trainCNN(self):

        netDataPath = self._getExtraPath("nnetData")
        makePath(netDataPath)
        nEpochs = self.nEpochs.get()
            
        posTrainDict = {self._getExtraPath("particles_AND.xmd"):  1}
        negTrainDict = {self._getExtraPath("particles_NOISE.xmd"):  1}

        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_PRECOMP:
          negTrainDict[self._getTmpPath("addNegTrainParticles.xmd")]= 1
        
        if self.usesGpu():
          numberOfThreads = None
          gpuToUse = self.getGpuList()[0]
        else:
          numberOfThreads = self.numberOfThreads.get()
          gpuToUse = None

        if self.trainTrueSetOfParticles.get():
          posTrainFn = self._getExtraPath("trainTrueParticlesSet.xmd")
          posTrainDict[posTrainFn] = self.trainPosWeight.get()
        if self.trainFalseSetOfParticles.get():
          negTrainFn = self._getExtraPath("trainFalseParticlesSet.xmd")
          negTrainDict[negTrainFn] = self.trainNegWeight.get()

        effectiveSize=-1
        if self._doContinue():
          prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
          copyTree(prevRunPath, netDataPath)
          if self.skipTraining.get():
              nEpochs = 0
        elif self._usePretrainedModel():
          nTrueParticles=  self._getEffectiveNumPartsTrain(posTrainDict)
          if nTrueParticles<1500:
            effectiveSize=1000
          elif 1500<=nTrueParticles<20000:
            effectiveSize=5000
          else:
            effectiveSize=50000
          self.__retrievePreTrainedModel(netDataPath, effectiveSize)
          if self.skipTraining.get():
              nEpochs = 0


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
        
    def predictCNN(self):

        netDataPath = self._getExtraPath('nnetData')
        if not os.path.isdir(netDataPath) and self._doContinue():
            prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
            copyTree(prevRunPath, netDataPath)
        elif self.skipTraining.get() and self._usePretrainedModel():
          self.__retrievePreTrainedModel(netDataPath)
            
        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        predictDict = {self._getExtraPath("particles_OR.xmd"): 1}
        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get():
            posTestDict = {self._getExtraPath("testTrueParticlesSet.xmd"): 1}
            negTestDict = {self._getExtraPath("testFalseParticlesSet.xmd"): 1}
        else:            
            posTestDict = None
            negTestDict = None
        outParticlesPath = self._getPath("particles.xmd")

        fnamesPred, weightsPred= self.__dataDict_toStrs(predictDict)
        
        args= " -n %s --mode score -i %s -o %s "%(netDataPath, fnamesPred, outParticlesPath)
                
        if posTestDict and posTestDict:
          fnamesPosTest, weightsPosTest= self.__dataDict_toStrs(posTestDict)
          fnamesNegTest, weightsNegTest= self.__dataDict_toStrs(posTestDict)
          args+= " --testingTrue %s --testingFalse %s "%(fnamesPosTest, fnamesNegTest)
          
        if not gpuToUse is None:
          args+= " -g %s"%(gpuToUse)
        if not numberOfThreads is None:
          args+= " -t %s"%(numberOfThreads)
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1)
                
    def createOutputStep(self):
        # PARTICLES
        cleanPattern(self._getPath("*.sqlite"))
        partSet = self._createSetOfParticles("outputParts_tmp")
        readSetOfParticles(self._getPath("particles.xmd"), partSet)
        inputSampling = self.inputCoordinates[0].get().getMicrographs().getSamplingRate()
        partSet.setSamplingRate(self._getDownFactor() * inputSampling)
        boxSize = self._getBoxSize()
        
        parSetCorrected= self._createSetOfParticles()
        parSetCorrected.copyInfo(partSet)
        # COORDINATES
        if not "OR" in self.coordinatesDict:
            self.loadCoords(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR'),'OR')

        coordSet = self._createSetOfCoordinates(
            self.coordinatesDict['OR'].getMicrographs())
        coordSet.copyInfo(self.coordinatesDict['OR'])
        coordSet.setBoxSize(boxSize)

        downFactor = self._getDownFactor()
        for part in partSet:
            coord = part.getCoordinate().clone()
            coord.scale(downFactor)
            deepZscoreLabel = '_xmipp_%s' % xmipp.label2Str(MD.MDL_ZSCORE_DEEPLEARNING1)
            setattr(coord, deepZscoreLabel, getattr(part, deepZscoreLabel))
            part = part.clone()
            part.scaleCoordinate(downFactor)
            coordSet.append(coord)
            parSetCorrected.append(part)
            
        coordSet.write()
        parSetCorrected.write()
        cleanPattern(self._getPath("particles.xmd"))    
        cleanPattern(self._getPath("*outputParts_tmp.sqlite"))
        writeSetOfParticles(parSetCorrected, self._getPath("particles.xmd") )
        self._defineOutputs(outputCoordinates=coordSet)

        for inSetOfCoords in self.inputCoordinates:
            self._defineSourceRelation(inSetOfCoords.get(), coordSet)

    def _summary(self):
        message = []
        for i, coordinates in enumerate(self.inputCoordinates):
            protocol = self.getMapper().getParent(coordinates.get())
            message.append("Data source  %d %s" % (i + 1, protocol.getClassLabel()))
        message.append("Relative Radius = %f" % self.consensusRadius)
        message.append("\nThe output contains the OR junction of all the input "
                       "coordinates with a 'zScoreDeepLearning1' value attached.\n"
                       "Please, click on 'Analyze Results' to make a subset.")
        return message

    def _methods(self):
        return []


    #--------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, MD.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(MD.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True


class XmippProtDeepConsSubSet(ProtUserSubSet):
    """ Create subsets from the GUI for the Deep Consensus protocol.
        This protocol will be executed mainly calling the script 'pw_create_image_subsets.py'
        from the ShowJ gui. The enabled/disabled changes will be stored in a temporary sqlite
        file that will be read to create the new subset.
        """

    def __init__(self, **args):
        ProtUserSubSet.__init__(self, **args)

    def _createSimpleSubset(self, inputObj):
        modifiedSet = inputObj.getClass()(filename=self._dbName,
                                          prefix=self._dbPrefix)

        className = inputObj.getClassName()
        createFunc = getattr(self, '_create' + className)
        output = createFunc(inputObj.getMicrographs())

        for item in modifiedSet:
            if item.isEnabled():
                coord = item.getCoordinate().clone()
                coord.scale(1)
                output.append(coord)

        output.copyInfo(inputObj)
        output.setBoxSize(inputObj.getBoxSize())

        # Register outputs
        self._defineOutput(className, output)

        if inputObj.hasObjId():
            self._defineTransformRelation(inputObj, output)

        return output
