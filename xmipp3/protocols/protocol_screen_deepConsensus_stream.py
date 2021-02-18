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
import os, sys
from glob import glob
import six
import json, shutil

from pyworkflow import VERSION_2_0
from pyworkflow.utils.path import (makePath, cleanPattern, cleanPath, copyTree,
                                   createLink)
from pwem.constants import RELATION_CTF, ALIGN_NONE
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfParticles, Set, SetOfMicrographs, Micrograph, SetOfCoordinates
from pwem.protocols import ProtParticlePicking, ProtUserSubSet

import pyworkflow.utils as pwutils
from pyworkflow.protocol import params, STATUS_NEW
import pwem.emlib.metadata as md

from pwem import emlib
import xmipp3
from xmipp3 import XmippProtocol
from xmipp3.protocols.protocol_pick_noise import pickNoise_prepareInput, IN_COORDS_POS_DIR_BASENAME
from xmipp3.convert import (readSetOfParticles, setXmippAttributes,
                            micrographToCTFParam, writeSetOfParticles,
                            writeSetOfCoordinates, readSetOfCoordsFromPosFnames, readSetOfCoordinates)

MIN_NUM_CONSENSUS_COORDS = 256

try:
  from xmippPyModules.deepConsensusWorkers.deepConsensus_networkDef import DEEP_PARTICLE_SIZE
except ImportError as e:
  DEEP_PARTICLE_SIZE = 128
  
class XmippProtScreenDeepConsensusStream(ProtParticlePicking, XmippProtocol):
    """ Protocol to compute a smart consensus between different particle picking
        algorithms. The protocol takes several Sets of Coordinates calculated
        by different programs and/or different parameter settings. Let's say:
        we consider N independent pickings. Then, a neural network is trained
        using different subset of picked and not picked cooridantes. Finally,
        a coordinate is considered to be a correct particle according to the
        neural network predictions.
    """
    _label = 'stream deep consensus picking'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_DLTK_v0.3'
    _stepsCheckSecs = 10              # time in seconds to check the steps

    CONSENSUS_COOR_PATH_TEMPLATE="consensus_coords_%s"
    CONSENSUS_PARTS_PATH_TEMPLATE="consensus_parts_%s"
    PRE_PROC_MICs_PATH="preProcMics"

    ADD_DATA_TRAIN_TYPES = ["None", "Precompiled", "Custom"]
    ADD_DATA_TRAIN_NONE = 0
    ADD_DATA_TRAIN_PRECOMP = 1
    ADD_DATA_TRAIN_CUST = 2

    ADD_DATA_TRAIN_CUSTOM_OPT = ["Particles", "Coordinates" ]
    ADD_DATA_TRAIN_CUSTOM_OPT_PARTS = 0
    ADD_DATA_TRAIN_CUSTOM_OPT_COORS = 1

    ADD_MODEL_TRAIN_TYPES = ["New", "Pretrained", "PreviousRun"]
    ADD_MODEL_TRAIN_NEW = 0
    ADD_MODEL_TRAIN_PRETRAIN = 1
    ADD_MODEL_TRAIN_PREVRUN = 2

    ADDED_DATA = False

    #Streaming parameters
    PREPROCESSED_MICFNS = []
    PREPROCESSING = False

    TO_EXTRACT_MICFNS = {'OR': [],
                         'NOISE': [],
                         'AND': []}
    EXTRACTING = {'OR': False,
                  'NOISE': False,
                  'AND': False}
    EXTRACTED_MICFNS = {'OR': [],
                        'NOISE': [],
                        'AND': [],
                        'ADDITIONAL_COORDS_TRUE': [],
                        'ADDITIONAL_COORDS_FALSE': []}

    TRAIN_BATCH_MIN = 5
    TRAIN_BATCH_MAX = 20
    TRAINED_MICFNS = []
    TO_TRAIN_MICFNS = []
    TRAINING = False
    TRAINING_PASS = 0
    TRAINED = False
    numPosTrainedParts = 0

    PREDICTED = []
    LAST_ROUND = False
    ENDED = False

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

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
                            
        form.addParallelSection(threads=4, mpi=1)
        
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
        form.addParam('threshold', params.FloatParam, default=0.5,
                      label='Tolerance threshold',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='The method attach a score between 0 and 1, where 0 '
                           'if for _bad_ particles and 1 for _good_ ones. '
                           'Introduce -1 to let pass all for posterior inspection.')

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
                      label='Did you invert the micrographs contrast (particles are bright now)?',
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
        form.addParam('toTrainDataSize', params.IntParam,
                      label="Aproximate number of particles for training",
                      default=20000, expertLevel=params.LEVEL_ADVANCED,
                      help='Aproximate number of particles for training the CNN\n'
                           'It will determine the number of channels in the CNN\n'
                           'Usually, the more channels the better, but more training data is needed\n'
                           'Three CNN sizes: n < 1500 | 1500 <= n < 20000 | n >= 20000')
                           
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
                      help='If you set to *%s*, only the AND and RANDOM will be used for training.\n'
                           'If you set to *%s*, a precompiled additional training set will be added to '
                           'to the AND and RANDOM sets for training.\n'
                           'If you set to *%s*, you can provide your own data that will be added to '
                           'the AND and RANDOM sets for training.\n'
                            %tuple( self.ADD_DATA_TRAIN_TYPES)
                      )


        form.addParam('trainingDataType', params.EnumParam,
                      condition=("(modelInitialization==%s or not skipTraining ) " +
                       "and addTrainingData==%s") % (self.ADD_MODEL_TRAIN_NEW, self.ADD_DATA_TRAIN_CUST),
                      choices=self.ADD_DATA_TRAIN_CUSTOM_OPT,
                      default=self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS,
                      label='Additional training data',
                      help='You can provide either particles or coordinates as additional training set.' \
                           'If you provide coordinantes, they have to be picked from the same micrographs that the' \
                           'inputs\n If you provide particles, they have to be processed in the same way that the protocol' \
                           'does (128x128 pixels and withe particles). Thus, what the protocol does is to perform the ' \
                           'following steps:\n'
                            '1) mic donwsampling to the required size such that '
                            'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                            ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                            '2) mic normalization to 0 mean and 1 std and mic contrast inversion to have WHITE particles.\n '
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
                            'preprocessed as indicated before.\n' )

        form.addParam('trainTrueSetOfParticles', params.PointerParam,
                      label="Positive train particles 128px (optional)",
                      pointerClass='SetOfParticles', allowsNull=True,
                      condition=("(modelInitialization==%s or not skipTraining ) "+
                                "and addTrainingData==%s and trainingDataType==%s")%(self.ADD_MODEL_TRAIN_NEW,
                                                                                     self.ADD_DATA_TRAIN_CUST,
                                                                                     self.ADD_DATA_TRAIN_CUSTOM_OPT_PARTS),
                      help='Select a set of true positive particles. '
                           'Take care of the preprocessing (128x128 pixels, contrast inverted (white particles), possibly '
                           'CTF corrected')


        form.addParam('trainTrueSetOfCoords', params.PointerParam,
                      label="Positive coordinates(optional)",
                      pointerClass='SetOfCoordinates', allowsNull=True,
                      condition="(modelInitialization==%s or not skipTraining ) "
                                "and addTrainingData==%s and trainingDataType==%s"
                                % (self.ADD_MODEL_TRAIN_NEW,
                                   self.ADD_DATA_TRAIN_CUST,
                                   self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS),
                      help="Select a set of true coordinates collected from the "
                           "same microgaphs that the input")

        form.addParam('trainPosWeight', params.IntParam, default='1',
                      label="Weight of positive additional train data",
                      condition="(modelInitialization==%s or not skipTraining ) "
                                "and addTrainingData==%s"
                                % (self.ADD_MODEL_TRAIN_NEW,
                                   self.ADD_DATA_TRAIN_CUST),
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
                      condition="(modelInitialization==%s or not skipTraining ) "
                                "and addTrainingData==%s and trainingDataType==%s"
                                % (self.ADD_MODEL_TRAIN_NEW,
                                   self.ADD_DATA_TRAIN_CUST,
                                   self.ADD_DATA_TRAIN_CUSTOM_OPT_PARTS),
                      help='Select a set of false positive particles. '
                           'Take care of the preprocessing: 128x128 pixels, '
                           'contrast inverted (white particles), '
                           'possibly CTF corrected')

        form.addParam('trainFalseSetOfCoords', params.PointerParam,
                      label="Negative coordinates(optional)",
                      pointerClass='SetOfCoordinates', allowsNull=True,
                      condition="(modelInitialization==%s or not skipTraining ) "
                                "and addTrainingData==%s and trainingDataType==%s"
                                % (self.ADD_MODEL_TRAIN_NEW,
                                   self.ADD_DATA_TRAIN_CUST,
                                   self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS),
                      help="Select a set of incorrect coordinates collected from "
                           "the same microgaphs that the input")

        form.addParam('trainNegWeight', params.IntParam, default='1',
                      label="Weight of negative additional train data",
                      condition="(modelInitialization==%s or not skipTraining ) "
                                "and addTrainingData==%s"
                                % (self.ADD_MODEL_TRAIN_NEW,
                                   self.ADD_DATA_TRAIN_CUST,),
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
        errorMsg = self.validateDLtoolkit(errorMsg, model="deepConsensus",
                                          assertModel=self.addTrainingData.get()==self.ADD_DATA_TRAIN_PRECOMP)
        return errorMsg

#--------------------------- INSERT steps functions ---------------------------

    def _doContinue(self):
        return self.modelInitialization.get()== self.ADD_MODEL_TRAIN_PREVRUN

    def justPredict(self):
      return self.skipTraining.get()==True

    def _usePretrainedModel(self):
        return self.modelInitialization.get()== self.ADD_MODEL_TRAIN_PRETRAIN

    def endProtocolStep(self):
      self.lastTrainingPass = self.TRAINING_PASS
      self.TRAINING_PASS = ''
      self.retrievePreviousPassModel()
      depPredict = self._insertFunctionStep('predictCNN')
      self._insertFunctionStep('createOutputStep', prerequisites=[depPredict])
      self.ENDED = True


    def _insertAllSteps(self):
        self.inputMicrographs = None
        self.boxSize = None
        self.coordinatesDict = {}

        self.initDeps = [self._insertFunctionStep("initializeStep")]
        self.lastStep = self._insertFunctionStep('lastRoundStep', wait=True, prerequisites=self.initDeps)
        self.endStep = self._insertFunctionStep('endProtocolStep', wait=True, prerequisites=[self.lastStep])   # finish the protocol


    def _stepsCheck(self):
        newSteps = []
        if not self.ENDED:
          preprocMicsPath = self._getTmpPath(self.PRE_PROC_MICs_PATH)
          outCoordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR')
          # Functions streamed. Input is processed as soon as it is produced
          if not self.checkIfPrevRunIsCompatible("mics_") and len(self.readyToPreprocessMics()) > 0 and not self.PREPROCESSING:
            self.lastDeps = [self._insertFunctionStep("preprocessMicsStep", prerequisites=self.initDeps)]
            self.PREPROCESSING = True
          if self.checkIfNewMics('OR') and not self.EXTRACTING['OR']:
            newSteps += [self.insertCaculateConsensusSteps('OR', prerequisites=self.initDeps)]
            newSteps += self.insertExtractPartSteps('OR', prerequisites=newSteps)
            self.EXTRACTING['OR'] = True
          if not self.skipTraining.get():
            if self.checkIfNewMics('NOISE') and not self.EXTRACTING['NOISE']:
              depNoise = self._insertFunctionStep('pickNoise', prerequisites=self.initDeps)
              depsNoise = self.insertExtractPartSteps('NOISE', prerequisites=[depNoise])
              newSteps += depsNoise
              self.EXTRACTING['NOISE'] = True

            if self.checkIfNewMics('AND') and not self.EXTRACTING['AND']:
              depsAnd = self.insertCaculateConsensusSteps('AND', prerequisites=self.initDeps)
              depsAnd = self.insertExtractPartSteps('AND', prerequisites=[depsAnd])
              newSteps += depsAnd
              self.EXTRACTING['AND'] = True

            if self.addTrainingData.get() == self.ADD_DATA_TRAIN_CUST and \
                    self.trainingDataType == self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS and not self.ADDED_DATA:
              if self.trainTrueSetOfCoords.get() is not None:
                newSteps += self.insertExtractPartSteps('ADDITIONAL_COORDS_TRUE', prerequisites=self.initDeps)
                self.ADDED_DATA = True
              if self.trainFalseSetOfCoords.get() is not None:
                newSteps += self.insertExtractPartSteps('ADDITIONAL_COORDS_FALSE', prerequisites=self.initDeps)
                self.ADDED_DATA = True

            if not self.TRAINING and self.numPosTrainedParts < self.toTrainDataSize.get():
              self.TO_TRAIN_MICFNS = self.readyToTrainMicFns()
              if len(self.TO_TRAIN_MICFNS) >= self.TRAIN_BATCH_MIN:
                self.depsTrain = [self._insertFunctionStep('trainCNN', self.TO_TRAIN_MICFNS, prerequisites=self.initDeps)]
                newSteps += self.depsTrain
          else:
            self.TRAINED = True
            self.PREDICTED = []

          if self.TRAINED and self.TRAINING_PASS not in self.PREDICTED:
            depPredict = self._insertFunctionStep('predictCNN', prerequisites= self.depsTrain)
            newSteps += [self._insertFunctionStep('createOutputStep', prerequisites=[depPredict])]
            self.PREDICTED.append(self.TRAINING_PASS)

          if not self.checkIfNewMics() and self.TRAINED and not self.TRAINING and not self.LAST_ROUND:
            # Finiquitar el protocolo porque no hay nuevas micrografias
            protLast = self._steps[self.lastStep - 1]
            protLast.addPrerequisites(*newSteps)
            protLast.setStatus(STATUS_NEW)
            self.updateSteps()

          protEnd = self._steps[self.endStep-1]
          protEnd.addPrerequisites(*newSteps)
          if self.LAST_ROUND:
            protEnd.setStatus(STATUS_NEW)
          self.updateSteps()

    def lastRoundStep(self):
      self.TRAIN_BATCH_MIN = 1
      self.LAST_ROUND = True

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
          if self.trainingDataType== self.ADD_DATA_TRAIN_CUSTOM_OPT_PARTS:
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


    def checkIfNewMics(self, mode=''):
      preprocMicsPath = self._getTmpPath(self.PRE_PROC_MICs_PATH)
      if mode == '':
        for mode in ['OR','NOISE','AND']:
          if len(self.readyToExtractMicFns(mode)) > 0:
            return True
      else:
        if len(self.readyToExtractMicFns(mode)) > 0:
          return True
      return False

    def readyToExtractMicFns(self, mode):
      return list(set(self.PREPROCESSED_MICFNS) -
                  (set(self.EXTRACTED_MICFNS[mode]) | set(self.TO_EXTRACT_MICFNS[mode])))

    def readyToTrainMicFns(self):
      extractedMicFns = set(self.EXTRACTED_MICFNS['OR']) & set(self.EXTRACTED_MICFNS['NOISE']) & \
                        set(self.EXTRACTED_MICFNS['AND'])
      readyToTrain = list(extractedMicFns - set(self.TRAINED_MICFNS))
      return readyToTrain[:min(len(readyToTrain), self.TRAIN_BATCH_MAX)]

    def readyToPreprocessMics(self):
      micFns = self.getInputMicsFns()
      return list(set(micFns) - set(self.PREPROCESSED_MICFNS))

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
        modelPath = self.getModel("deepConsensus", modelType)
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
        outCoordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE% mode)
        if not os.path.exists(outCoordsDataPath):
          makePath(outCoordsDataPath)
        newDep = self._insertFunctionStep('calculateCoorConsensusStep', outCoordsDataPath, mode, prerequisites=prerequisites)
        newDep = self._insertFunctionStep('loadCoords', outCoordsDataPath, mode, prerequisites=[newDep])
        return newDep


    def calculateCoorConsensusStep(self, outCoordsDataPath, mode):
      #Solo calcular el consenso e las coordenadas que no han sido ya extraidas
      if self.checkIfNewMics(mode):
        extractedSetOfCoordsFns = []
        for micFn in self.EXTRACTED_MICFNS[mode]:
          extractedSetOfCoordsFns.append(micFn.replace('mrc','pos'))

        Tm = []
        for coordinatesP in self.inputCoordinates:
          mics = coordinatesP.get().getMicrographs()
          Tm.append(mics.getSamplingRate())
        nCoordsSets= len(Tm)

        inputCoordsFnames={}
        for coord_num, coordinatesP in enumerate(self.inputCoordinates):
            tmpPosDir= self._getTmpPath("input_coords_%d"%(coord_num))
            if not os.path.exists(tmpPosDir):
              makePath(tmpPosDir)
            writeSetOfCoordinates(tmpPosDir, coordinatesP.get(), scale=float(Tm[coord_num])/float(Tm[0]))
            for posFname in os.listdir(tmpPosDir):
                baseName, extension=os.path.splitext(os.path.basename(posFname))
                if extension==".pos" and not posFname in extractedSetOfCoordsFns:
                  if baseName not in inputCoordsFnames:
                      inputCoordsFnames[baseName]=["None"]*nCoordsSets
                  inputCoordsFnames[baseName][coord_num]= os.path.join(tmpPosDir, posFname)
        inputFileHeader="#pos_i\n"
        inputFileStr=inputFileHeader
        for baseName in inputCoordsFnames:
          fnames= inputCoordsFnames[baseName]
          inputFileStr+=" ".join(fnames)+"\n"

        assert len(inputFileStr)>len(inputFileHeader), "Error, no consensus can be computed as there " \
                                                       "are mismatch in coordinate sets filenames"
        consensus = -1 if mode=="AND" else 1
        configFname= self._getTmpPath("consensus_%s_inputs.txt"%(mode) )
        with open(configFname, "w") as f:
            f.write(inputFileStr)

        args="-i %s -s %d -c %d -d %f -o %s -t %d"%(configFname, self._getBoxSize(), consensus, self.consensusRadius.get(),
                                                   outCoordsDataPath, self.numberOfThreads.get())
        self.runJob('xmipp_coordinates_consensus', args, numberOfMpi=1)
        self.TO_EXTRACT_MICFNS[mode] = self.readyToExtractMicFns(mode)


    def loadCoords(self, posCoorsPath, mode, micSet=[]):
        #Upload coords sqlite
        sqliteName= self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode)+".sqlite"
        if os.path.isfile(self._getExtraPath(sqliteName)):
            cleanPath(self._getExtraPath(sqliteName))
        totalSetOfCoordinates= readSetOfCoordsFromPosFnames(posCoorsPath, setOfInputCoords= self.inputCoordinates[0].get(),
                                    sqliteOutName= sqliteName, write=True )
        if not micSet == []:
          #Load coordinates from an specific set of mics
          batchSetOfCoordinates = self._createSetOfCoordinates(micSet)
          batchSetOfCoordinates.setBoxSize(self._getBoxSize())
          readSetOfCoordinates(posCoorsPath, micSet=micSet, coordSet = batchSetOfCoordinates)
          if mode in self.coordinatesDict:
            for newCoord in batchSetOfCoordinates:
              try:
                self.coordinatesDict[mode].append(newCoord)
              except:
                pass
          else:
            self.coordinatesDict[mode] = batchSetOfCoordinates
        else:
          print("Coordinates %s size: %d" % (mode, totalSetOfCoordinates.getSize()))
          assert totalSetOfCoordinates.getSize() > MIN_NUM_CONSENSUS_COORDS, \
            ("Error, the consensus (%s) of your input coordinates was too small (%s). " +
             "It must be > %s. Try a different input..."
             ) % (mode, str(totalSetOfCoordinates.getSize()), str(MIN_NUM_CONSENSUS_COORDS))
          self.coordinatesDict[mode] = totalSetOfCoordinates

    def pickNoise(self):
        if self.checkIfPrevRunIsCompatible("coords_"):
          print("using previous round noise particles")
        else:
          orPosDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % "OR")
          outputPosDir= self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % "NOISE")
          if not "OR" in self.coordinatesDict:  # fill self.coordinatesDict['OR']
              self.loadCoords( self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR'),
                               'OR')
          #Getting the extracted mics where noise have not been picked yet
          micsDir = self._getTmpPath(self.PRE_PROC_MICs_PATH)
          toPickNoiseFns = list(set(self.EXTRACTED_MICFNS['OR']) -
                                (set(self.EXTRACTED_MICFNS['NOISE']) | set(self.TO_EXTRACT_MICFNS['NOISE'])))
          if len(toPickNoiseFns) > 0:
            micNoiseSet = self.loadMicSetFromFns(micsDir, toPickNoiseFns)
            toNoiseSetOfCoords = self._createSetOfCoordinates(micNoiseSet)
            toNoiseSetOfCoords.setBoxSize(self._getBoxSize())
            readSetOfCoordinates(orPosDir, micSet=micNoiseSet, coordSet = toNoiseSetOfCoords)

            # Write the tonoise mic files into a tmp directory
            coordsDir = self._getTmpPath(IN_COORDS_POS_DIR_BASENAME)
            toPickMicsDir = micsDir + '_toPickNoise'
            toPickCoorsDir = coordsDir + '_toPickNoise'
            if not os.path.exists(coordsDir):
              makePath(coordsDir)
            if os.path.exists(toPickMicsDir):
              cleanPath(toPickMicsDir)
            if os.path.exists(toPickCoorsDir):
              cleanPath(toPickCoorsDir)
            makePath(toPickMicsDir)
            makePath(toPickCoorsDir)
            writeSetOfCoordinates(coordsDir, toNoiseSetOfCoords)

            for micFn in toPickNoiseFns:
              posFn = pwutils.path.replaceBaseExt(micFn, 'pos')
              shutil.copyfile(micsDir + '/' + micFn, toPickMicsDir + '/' + micFn)
              shutil.copyfile(coordsDir + '/' + posFn, toPickCoorsDir + '/' + posFn)

            argsDict=pickNoise_prepareInput(toNoiseSetOfCoords, self._getTmpPath())
            argsDict['toPickMicsDir'] = toPickMicsDir
            argsDict['toPickCoorsDir'] = toPickCoorsDir
            argsDict["outputPosDir"]= outputPosDir
            argsDict["nThrs"] = self.numberOfThreads.get()
            argsDict["nToPick"]=-1
            args=(" -i %(toPickMicsDir)s -c %(toPickCoorsDir)s -o %(outputPosDir)s -s %(boxSize)s "+
                  "-n %(nToPick)s -t %(nThrs)s")%argsDict

            if not self.checkIfPrevRunIsCompatible( "coords_"):
                self.runJob('xmipp_pick_noise', args, numberOfMpi=1)

            self.loadCoords(outputPosDir, 'NOISE', micSet= micNoiseSet)
            self.TO_EXTRACT_MICFNS['NOISE'] = toPickNoiseFns
            print('Adding to extract {} {} micrographs'.format(len(toPickNoiseFns), 'NOISE'))


    def loadMicSetFromFns(self, inputDir, micFns):
      micSet = self._createSetOfMicrographs()
      for micFn in micFns:
        micPath = os.path.join(inputDir, micFn)
        micSet.append(Micrograph(micPath))
      micSet.copyInfo(self._getInputMicrographs())
      return micSet


    def getPreProcParamsFromForm(self):
        mics_= self._getInputMicrographs()
        mic = mics_.getFirstItem()
        fnMic = mic.getFileName()
        pathToMics= os.path.split(fnMic)[0]
        pathToCtfs= "None"

        if not self.ignoreCTF.get():
          pathToCtfs=  os.path.split(self.ctfRelations.get().getFileName())[0]

        paramsInfo={"mics_ignoreCTF": self.ignoreCTF.get(),
                    "mics_skipInvert": self.skipInvert.get(),
                    "mics_pathToMics": pathToMics,
                    "mics_pathToCtfs": pathToCtfs}
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

            preprocParams_loaded= {k: _makeTupleIfList(preprocParams_loaded[k])
                                   for k in preprocParams_loaded
                                   if k.startswith(inputType)}

            for key in preprocParams_loaded:
              if "path" in key:
                for val in preprocParams_loaded[key]:
                  if not val=="None" and not os.path.exists(val):
                    return False

            shared_items = {k: preprocParams_loaded[k]
                            for k in preprocParams_loaded
                            if k in preprocParams and preprocParams_loaded[k] == preprocParams[k]}

            return (len(shared_items) == len(preprocParams) and
                    len(shared_items) == len(preprocParams_loaded) and
                    len(shared_items) > 0)

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

    def getInputMicsFns(self):
      micFnames = []
      micIds = self.getMicsIds(filterOutNoCoords=True)
      mics = self._getInputMicrographs()
      for micId in micIds:
        micFnames.append(mics[micId].getFileName().split('/')[-1])
      return sorted(micFnames)

    def preprocessMicsStep(self):
        micIds = self.getMicsIds(filterOutNoCoords=True)
        if len(micIds) > 0:
          samplingRate = self._getInputMicrographs().getSamplingRate()
          mics_ = self._getInputMicrographs()
          micsFnameSet = {mics_[micId].getMicName(): mics_[micId].getFileName() for micId in micIds
                          if mics_[micId] is not None}  # to skip failed mics
          toPreprocessMicFns = self.readyToPreprocessMics()
          self.PREPROCESSED_MICFNS += toPreprocessMicFns
          print('New mics being preprocessed: ', toPreprocessMicFns)
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
        self.PREPROCESSING = False

    def insertExtractPartSteps(self, mode, prerequisites):
        newSteps = []
        if not self.checkIfPrevRunIsCompatible(""):
          preprocMicsPath = self._getTmpPath(self.PRE_PROC_MICs_PATH)
          outCoordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode)

          newSteps.append(self._insertFunctionStep("extractParticles", mode, prerequisites= prerequisites))
          newSteps.append(self._insertFunctionStep("joinSetOfParticlesStep", mode, prerequisites= newSteps))
        return newSteps

    def _getConsensusParticlesDir(self, mode):
      pathFun = self._getTmpPath if mode != "OR" else self._getExtraPath
      return pathFun(self.CONSENSUS_PARTS_PATH_TEMPLATE% mode)

    def extractParticles(self, mode):
        micsFnameSet = {}
        preprocMicsPath = self._getTmpPath(self.PRE_PROC_MICs_PATH)
        toExtractMicFns = self.TO_EXTRACT_MICFNS[mode]
        print('To extract in mode {}: {}'.format(mode, toExtractMicFns))
        if len(toExtractMicFns) > 0:
          for micFname in toExtractMicFns:
            micFnameBase= pwutils.removeExt(micFname)
            micFname= os.path.join(preprocMicsPath, micFname)
            micsFnameSet[micFnameBase]= micFname
          extractCoordsContent="#mics coords\n"
          posDir= self._getExtraPath( self.CONSENSUS_COOR_PATH_TEMPLATE%mode )
          if mode.startswith("ADDITIONAL_COORDS"):
            os.mkdir(posDir)
            if mode.endswith("TRUE"):
              coordSet= self.trainTrueSetOfCoords.get()
            elif mode.endswith("FALSE"):
              coordSet= self.trainFalseSetOfCoords.get()
            writeSetOfCoordinates(posDir, coordSet)

          for posFname in os.listdir(posDir):
            posNameBase=  pwutils.removeExt(posFname)
            posFname= os.path.join(posDir, posFname)
            if posNameBase in micsFnameSet:
              extractCoordsContent+= "%s particles@%s\n"%(micsFnameSet[posNameBase], posFname)
            else:
              pass
              #print("WARNING, no micFn for coords %s"%(posFname))

          inputsFname= self._getTmpPath("extractParticles_inputs_%s.txt"%mode)
          outputDir= self._getConsensusParticlesDir(mode)
          if not os.path.exists(outputDir):
            makePath(outputDir)
          nThrs= self.numberOfThreads.get()
          with open(inputsFname, "w") as f:
            f.write(extractCoordsContent)
          downFactor= self._getDownFactor()
          args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, DEEP_PARTICLE_SIZE, downFactor, outputDir, nThrs)

          self.runJob('xmipp_extract_particles', args, numberOfMpi=1)


    def writeSetOfParticlesXmd(self, mode, micFns='', trainingPass=''):
        # Create images.xmd metadata joining from different .stk
        fnImages = self._getExtraPath("particles_{}{}.xmd".format(mode, trainingPass))
        posDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode)

        if micFns == '':
          micFns = self.TO_EXTRACT_MICFNS[mode]

        imgsXmd = md.MetaData()
        for micFn in micFns:
          posFn = os.path.join(posDir, pwutils.replaceBaseExt(micFn, "pos"))
          xmdFn = os.path.join(self._getConsensusParticlesDir(mode),
                               pwutils.replaceBaseExt(posFn, "xmd"))
          if os.path.exists(xmdFn):
            mdFn = md.MetaData(xmdFn)
            mdPos = md.MetaData('particles@%s' % posFn)
            mdPos.merge(mdFn)
            imgsXmd.unionAll(mdPos)
          else:
            self.warning("The coord file %s wasn't used for extraction! "
                         % os.path.basename(posFn))
        if not os.path.exists(fnImages):
          imgsXmd.write(fnImages)
        else:
          imgsXmd.append(fnImages)
        print('\nSet of {} particles written in {}'.format(mode, fnImages))

    def joinSetOfParticlesStep( self, mode):
        #Generalized version in: writeSetOfParticlesXmd
        #Create images.xmd metadata joining from different .stk
        fnImages = self._getExtraPath("particles_%s.xmd" % mode)
        toExtractMicFns = self.TO_EXTRACT_MICFNS[mode]
        if len(self.TO_EXTRACT_MICFNS[mode]) > 0:
          self.EXTRACTED_MICFNS[mode] += self.TO_EXTRACT_MICFNS[mode]
          self.TO_EXTRACT_MICFNS[mode] = []
          imgsXmd = md.MetaData()
          #posFiles = glob(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode, '*.pos'))
          posDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode)

          for micFn in toExtractMicFns:
            posFn = os.path.join(posDir, pwutils.replaceBaseExt(micFn, "pos"))
            xmdFn = os.path.join(self._getConsensusParticlesDir(mode),
                                       pwutils.replaceBaseExt(posFn, "xmd"))
            if os.path.exists(xmdFn):
              mdFn = md.MetaData(xmdFn)
              mdPos = md.MetaData('particles@%s' % posFn)
              mdPos.merge(mdFn)
              imgsXmd.unionAll(mdPos)
            else:
              self.warning("The coord file %s wasn't used for extraction! "
                           % os.path.basename(posFn))
          if not os.path.exists(fnImages):
            imgsXmd.write(fnImages)
          else:
            imgsXmd.append(fnImages)
          print('\nParticles xmd file {} updated'.format(fnImages))
        self.EXTRACTING[mode] = False


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
          mdObject = md.MetaData(mdPath)
          nParts+= mdObject.size()
        return nParts

    def retrievePreviousPassModel(self):
      trPass = self.TRAINING_PASS
      curNetDataPath = self._getExtraPath("nnetData{}".format(trPass))
      if trPass == '':
        prevNetDataPath = self._getExtraPath("nnetData{}".format(self.lastTrainingPass))
      else:
        prevNetDataPath = self._getExtraPath("nnetData{}".format(trPass - 1))
      copyTree(prevNetDataPath, curNetDataPath)


    def __retrievePreTrainedModel(self, netDataPath, effectiveSize=-1):
        if effectiveSize==-1:
          effectiveSize=int(5e4)
        modelTypeDir= "keras_models/%sPhaseFlip_Invert/nnetData_%d/tfchkpoints_0" % (
                            "no" if self.ignoreCTF.get() else "", effectiveSize)
        modelTypeDir= self.getModel("deepConsensus", modelTypeDir)

        for i in range(self.nModels.get()):
          targetPath= os.path.join(netDataPath, "tfchkpoints_%d"%(i))
          print(targetPath, modelTypeDir)
          copyTree(modelTypeDir, targetPath)

    def trainCNN(self, toTrainMicFns):
        self.TRAINING = True
        if not self.TRAINING_PASS == '':
          self.TRAINING_PASS += 1
        #Writting the inputs in xmd
        for mode in ['AND', 'NOISE', 'OR']:
          self.writeSetOfParticlesXmd(mode, toTrainMicFns, self.TRAINING_PASS)
        #Creatting the training pass directory
        netDataPath = self._getExtraPath("nnetData{}".format(self.TRAINING_PASS))
        if not os.path.exists(netDataPath):
          makePath(netDataPath)
        nEpochs = self.nEpochs.get()

        #Setting the input and input weights
        posTrainDict = {self._getExtraPath("particles_AND{}.xmd".format(self.TRAINING_PASS)):  1}
        negTrainDict = {self._getExtraPath("particles_NOISE{}.xmd".format(self.TRAINING_PASS)):  1}
        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_PRECOMP and self.TRAINING_PASS==1:
          negTrainDict[self._getTmpPath("addNegTrainParticles.xmd")]= 1

        if self.usesGpu():
          numberOfThreads = None
          gpuToUse = self.getGpuList()[0]
        else:
          numberOfThreads = self.numberOfThreads.get()
          gpuToUse = None

        print(self.trainTrueSetOfParticles.get(), self.trainFalseSetOfParticles.get(), self.trainTrueSetOfCoords.get(),
              self.trainFalseSetOfCoords.get() )

        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_CUST and self.TRAINING_PASS==1:
          if self.trainingDataType.get() == self.ADD_DATA_TRAIN_CUSTOM_OPT_PARTS:
            if self.trainTrueSetOfParticles.get():
              posTrainFn = self._getExtraPath("trainTrueParticlesSet.xmd")
              posTrainDict[posTrainFn] = self.trainPosWeight.get()
            if self.trainFalseSetOfParticles.get():
              negTrainFn = self._getExtraPath("trainFalseParticlesSet.xmd")
              negTrainDict[negTrainFn] = self.trainNegWeight.get()

          elif self.trainingDataType.get() == self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS:
            if self.trainTrueSetOfCoords.get():
              posTrainFn = self._getExtraPath("particles_ADDITIONAL_COORDS_TRUE.xmd")
              posTrainDict[posTrainFn] = self.trainPosWeight.get()
            if self.trainFalseSetOfCoords.get():
              negTrainFn = self._getExtraPath("particles_ADDITIONAL_COORDS_FALSE.xmd")
              negTrainDict[negTrainFn] = self.trainNegWeight.get()

        effectiveSize=-1
        nTrueParticles = self.toTrainDataSize.get()
        if self._doContinue():
          prevRunPath = self.continueRun.get()._getExtraPath('nnetData{}'.format(self.TRAINING_PASS))
          copyTree(prevRunPath, netDataPath)
          if self.skipTraining.get():
              nEpochs = 0
        elif self._usePretrainedModel():
          #nTrueParticles=  self._getEffectiveNumPartsTrain(posTrainDict)
          if nTrueParticles<1500:
            effectiveSize=1000
          elif 1500<=nTrueParticles<20000:
            effectiveSize=5000
          else:
            effectiveSize=50000
          self.__retrievePreTrainedModel(netDataPath, effectiveSize)
          if self.skipTraining.get():
              nEpochs = 0
        elif not self.TRAINING_PASS == '' and self.TRAINING_PASS > 1:
          #Starting from model in previous trainingPass
          self.retrievePreviousPassModel()

        self.numPosTrainedParts += self._getEffectiveNumPartsTrain(posTrainDict)
        fnamesPos, weightsPos= self.__dataDict_toStrs(posTrainDict)
        fnamesNeg, weightsNeg= self.__dataDict_toStrs(negTrainDict)
        args= " -n %s --mode train -p %s -f %s --trueW %s --falseW %s --effective_data_size %s"%(netDataPath,
                      fnamesPos, fnamesNeg, weightsPos, weightsNeg, nTrueParticles)
        args+= " -e %s -l %s -r %s -m %s "%(nEpochs, self.learningRate.get(), self.l2RegStrength.get(),
                                          self.nModels.get())

        if not self.auto_stopping.get():
          args+=" -s"

        if not gpuToUse is None:
          args+= " -g %s"%(gpuToUse)
        if not numberOfThreads is None:
          args+= " -t %s"%(numberOfThreads)
          
        self.TRAINED_MICFNS += self.TO_TRAIN_MICFNS
        self.TRAINING = False
        self.TRAINED = True
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1, env=self.getCondaEnv())
        
    def predictCNN(self):
        netDataPath = self._getExtraPath("nnetData{}".format(self.TRAINING_PASS))
        if not os.path.isdir(netDataPath) and self._doContinue():
            prevRunPath = self.continueRun.get()._getExtraPath("nnetData{}".format(self.TRAINING_PASS))
            copyTree(prevRunPath, netDataPath)
        elif self.skipTraining.get() and self._usePretrainedModel():
          self.__retrievePreTrainedModel(netDataPath)

        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        mdORPath = self._getExtraPath("particles_OR.xmd")
        mdObject = md.MetaData(mdORPath)
        print('Predicting on {} true particles'.format(mdObject.size()))
        predictDict = {mdORPath: 1}
        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get():
            posTestDict = {self._getExtraPath("testTrueParticlesSet.xmd"): 1}
            negTestDict = {self._getExtraPath("testFalseParticlesSet.xmd"): 1}
        else:
            posTestDict = None
            negTestDict = None
        outParticlesPath = self._getPath("particles{}.xmd".format(self.TRAINING_PASS))

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
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1,
                    env=self.getCondaEnv())
                
    def createOutputStep(self):
        # PARTICLES
        trPass = self.TRAINING_PASS
        #cleanPattern(self._getPath("*.sqlite"))
        partSet = self._createSetOfParticles("outputParts_tmp{}".format(trPass))
        readSetOfParticles(self._getPath("particles{}.xmd".format(trPass)), partSet)
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
            deepZscoreLabel = '_xmipp_%s' % emlib.label2Str(md.MDL_ZSCORE_DEEPLEARNING1)
            setattr(coord, deepZscoreLabel, getattr(part, deepZscoreLabel))
            part = part.clone()
            part.scaleCoordinate(downFactor)
            if (self.threshold.get() < 0 or
                    getattr(part, deepZscoreLabel) > self.threshold.get()):
                coordSet.append(coord)
                parSetCorrected.append(part)

        coordSet.write()
        parSetCorrected.write()
        cleanPattern(self._getPath("particles{}.xmd".format(trPass)))
        cleanPattern(self._getPath("*outputParts_tmp{}.sqlite".format(trPass)))
        writeSetOfParticles(parSetCorrected, self._getPath("particles{}.xmd".format(trPass)) )
        self._defineOutputs(outputCoordinates=coordSet, outputParticles=parSetCorrected)

        for inSetOfCoords in self.inputCoordinates:
            self._defineSourceRelation(inSetOfCoords.get(), coordSet)
        self._store(coordSet, parSetCorrected)

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
        setXmippAttributes(item, row, md.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(md.MDL_ENABLED) <= 0:
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
