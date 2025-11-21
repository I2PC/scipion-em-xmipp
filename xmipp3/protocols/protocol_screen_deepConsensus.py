# **************************************************************************
# *
# * Authors:    Ruben Sanchez Garcia (rsanchez@cnb.csic.es)
# *             David Maluenda (dmaluenda@cnb.csic.es)
# *             Daniel Del Hoyo (daniel.delhoyo.gomez@alumnos.upm.es)
# *             Daniel Marchán (da.marchan@cnb.csic.es)
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
import os, sys, time
from glob import glob
import six
import json, shutil, pickle

from pyworkflow import VERSION_2_0
from pyworkflow.utils.path import (makePath, cleanPattern, cleanPath, copyTree,
                                   createLink)
from pwem.constants import RELATION_CTF, ALIGN_NONE
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfParticles, Micrograph, Particle, Coordinate
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
from pyworkflow import BETA, UPDATED, NEW, PROD


MIN_NUM_CONSENSUS_COORDS = 256
AND = 'by_all'
OR = 'by_at_least_one'
UNION_INTERSECTIONS = 'by_at_least_two'

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
        In streaming, the network is trained and used to predict in batches.
        The network is trained until the number of particles set is reached,
        meanwhile, a preliminary output is generated. Once the threshold is reached,
        the final output is produced by batches.
    """
    _label = 'deep consensus picking'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_DLTK_v0.3'
    _stepsCheckSecs = 5              # time in seconds to check the steps
    _devStatus = PROD


    USING_INPUT_COORDS = False
    USING_INPUT_MICS = False
    CONSENSUS_COOR_PATH_TEMPLATE="consensus_coords_%s"
    CONSENSUS_PARTS_PATH_TEMPLATE="consensus_parts_%s"
    PRE_PROC_MICs_PATH="preProcMics"

    PARTICLES_TEMPLATE = "particles{}.xmd"
    NET_TEMPLATE = "nnetData{}"

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

    #Streaming parameters
    PREPROCESSING = False
    TO_EXTRACT_MICFNS = {'OR': [],
                         'NOISE': [],
                         'AND': [],
                         'ADDITIONAL_COORDS_TRUE':[],
                         'ADDITIONAL_COORDS_FALSE':[]}
    EXTRACTING = {'OR': False,
                  'NOISE': False,
                  'AND': False}

    TRAIN_BATCH_MAX = 20
    PREPROCESS_BATCH_MAX = 200
    PREDICT_BATCH_MAX = 20
    TO_TRAIN_MICFNS = []
    TRAINED_PARAMS_PATH = 'trainedParams.pickle'
    TRAINING = False
    PREDICTING = False

    LAST_ROUND = False
    ENDED = False

    counter = 0

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        #self.stepsExecutionMode = params.STEPS_PARALLEL

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
                            
        form.addParallelSection(threads=2, mpi=1)
        
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
        form.addParam('maxValAcc', params.FloatParam,
                      label="Training mean val_acc threshold", default=0.95,
                      condition="modelInitialization==%s or not skipTraining" % self.ADD_MODEL_TRAIN_NEW,
                      help='Stop training if at any training batch the selected threshold is achieved')
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
                      label="Expected number of particles to use for training", default=20000,
                      help='Number of particles for training the CNN. Once surpassed, there will not be more training\n'
                           'Set to -1 to use all the particles found\n'
                           'It will determine the size of the CNN'
                           'Usually, the bigger the better, but more training data is needed\n'
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

        form.addSection(label='Streaming')
        form.addParam('doPreliminarPredictions', params.BooleanParam, default=False,
                      label="Perform preliminar predictions with on training CNN",
                      help='The protocol will make preliminar preedictions with the network before it is fully trained\n'
                           'These preliminar results will be stored in a different output set')
        form.addParam('extractingBatch', params.IntParam, default='5',
                      label="Extraction batch size",
                      help='Size of the extraction batches (in number of micrographs)')
        form.addParam('trainingBatch', params.IntParam, default='5',
                      label="Training batch size",
                      help='Size of the training batches (in number of micrographs).'
                           'The CNN needs a minimum number of particles to train for each batch, if there are not'
                           ' enough particles, the batch size must be increased')


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
    
    def _useNewModel(self):
      return self.modelInitialization.get() == self.ADD_MODEL_TRAIN_NEW

    def _usePretrainedModel(self):
        return self.modelInitialization.get()== self.ADD_MODEL_TRAIN_PRETRAIN

    def _insertAllSteps(self):
        self.inputMicrographs = None
        self.boxSize = None
        self.coordinatesDict = {}

        self.initDeps = [self._insertFunctionStep("initializeStep")]
        self.lastStep = self._insertFunctionStep('lastRoundStep', wait=True, prerequisites=self.initDeps)
        self.endStep = self._insertFunctionStep('endProtocolStep', wait=True, prerequisites=[self.lastStep])

    def getGpusList(self, separator):
        strGpus = ""
        for elem in self._stepsExecutor.getGpuList():
            strGpus = strGpus + str(elem) + separator
        return strGpus[:-1]

    def setGPU(self, oneGPU=False):
        if oneGPU:
            gpus = self.getGpusList(",")[0]
        else:
            gpus = self.getGpusList(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.info(f'Visible GPUS: {gpus}')
        return gpus

    def _stepsCheck(self):
        '''Checks if new steps can be executed'''
        self.newSteps = []
        if self.ENDED:
          return
        # Preprocessing
        if len(self.readyToPreprocessMics(shared=False)) > 0 and not self.PREPROCESSING and (self.counter%2 == 0):
            #print('----------------------------------ENTERING PREPROCESSING STEP-----------------------------')
            self.PREPROCESSING = True
            self.lastDeps = [self._insertFunctionStep("preprocessMicsStep", prerequisites=self.initDeps)]

        #Particle extraction OR (for predictions)
        if len(self.readyToExtractMicFns('OR')) >= self.extractingBatch.get() and not self.EXTRACTING['OR']:
          #print('----------------------------ENTERING OR EXTRACTION FOR PREDICTION-----------------------------')
          self.EXTRACTING['OR'] = True
          self.newSteps += self.insertCaculateConsensusSteps('OR', prerequisites=self.initDeps)
          self.newSteps += self.insertExtractPartSteps('OR', prerequisites=self.newSteps)

        #Particle extraction for training and training
        trainedParams = self.loadTrainedParams()
        toTrainSize = self.toTrainDataSize.get() if self.toTrainDataSize.get() != -1 else 1e10
        if self.trainingOn() and trainedParams['posParticlesTrained'] < toTrainSize and trainedParams['trainingPass'] != '':
          #print('----------------------------------ENTERING AND EXTRACTION AND/OR TRAINING--------------------')
          self.doTraining()

        elif trainedParams['posParticlesTrained'] >= toTrainSize and trainedParams['trainingPass'] != '':
          #print('-------------------------ENTERING IN THE CHANGING THE TRAINING_PASS LOGIC-------------------------')
          lastTrainingPass = trainedParams['trainingPass']
          self.retrievePreviousPassModel('', lastTrainingPass)
          trainedParams['trainingPass'] = ''
          self.saveTrainedParams(trainedParams)

        #Prediction
        if self.networkReadyToPredict() and self.cnnFree() and self.predictionsOn() and len(self.readyToPredictMicFns()) > 0:
            #print('---------------------------------------------ENTERING PREDICTION---------------------------------')
            self.PREDICTING = True
            depPredict = self._insertFunctionStep('predictCNN', prerequisites= self.newSteps)
            self.newSteps += [self._insertFunctionStep('endPredictingStep', prerequisites=[depPredict])]
            self.newSteps += [self._insertFunctionStep('createOutputStep', prerequisites=[depPredict])]

        #Last round with batch size == 1 to include all input
        if self.allFree() and not self.LAST_ROUND and self.checkIfParentsFinished():
          #print('----------------------------------NOT LAST ROUND BUT ACTIVATES LAST_STEPS---------------------------')
          protLast = self._steps[self.lastStep - 1]
          protLast.addPrerequisites(*self.newSteps)
          protLast.setStatus(STATUS_NEW)
          self.updateSteps()

        protEnd = self._steps[self.endStep-1]
        protEnd.addPrerequisites(*self.newSteps)
        #Ending the protocol when everything is done
        if self.LAST_ROUND and self.allFree():
          #print('----------------------------------LAST ROUND ACTIVATES END_STEP---------------------------')
          protEnd.setStatus(STATUS_NEW)

        self.updateSteps()
        self.counter += 1

    def endPredictingStep(self):
      self.PREDICTING = False

    def endTrainingStep(self):
      self.saveTrainedParams(self.curTrainedParams)
      self.TRAINING = False
      mean_acc = self.loadMeanAccuracy()
      threshold = self.maxValAcc.get()
      trainedParams = self.loadTrainedParams()
      toTrainSize = self.toTrainDataSize.get() if self.toTrainDataSize.get() != -1 else 1e10

      if (mean_acc != None and mean_acc > threshold) or \
              (trainedParams['posParticlesTrained'] >= toTrainSize and trainedParams['trainingPass'] != ''):
        #print('-------------------------ENTERING IN THE CHANGING THE TRAININGPASS LOGIC-------------------------')
        lastTrainingPass = trainedParams['trainingPass']
        self.retrievePreviousPassModel('', lastTrainingPass)
        trainedParams['trainingPass'] = ''
        self.saveTrainedParams(trainedParams)
        if mean_acc > threshold:
            print('Mean accuracy %f surpass training accuracy threshold %f -> end training'
                %(mean_acc, threshold))

    def lastRoundStep(self):
      '''Starts the last round of training and predictions with the remainign microgrpahs
      when all the inputs have arrived'''
      self.extractingBatch.set(1)
      self.trainingBatch.set(1)
      self.LAST_ROUND = True

    def endProtocolStep(self):
      '''Finish the protocol with a final prediction using the final CNN'''
      lastTrainingPass = self.loadTrainedParams()['trainingPass']
      if lastTrainingPass != '':
        self.retrievePreviousPassModel('', lastTrainingPass)
        self.uploadTrainedParam('trainingPass', '')
        self.ENDED = True
        self.depLastPredict = self._insertFunctionStep('predictCNN', prerequisites=[self.endStep])
        self._insertFunctionStep('createOutputStep', True, prerequisites=[self.depLastPredict])
      else:
        self.updateOutput(closeStream=True)
        self.ENDED = True

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


        makePath(self._getExtraPath(self.PRE_PROC_MICs_PATH))
        for mode in ["AND", "OR", "NOISE"]:
            consensusCoordsPath = self.CONSENSUS_COOR_PATH_TEMPLATE % mode
            makePath(self._getExtraPath(consensusCoordsPath))

        if self._doContinue():
          if self.skipTraining.get():
            trPass=''
          else:
            trPass=1
          self.retrievePreviousRunModel(self.continueRun.get(), trPass)
          self.uploadTrainedParam('trainingPass', trPass)

        if self._usePretrainedModel() and self.skipTraining.get():
          trPass=''
          self.uploadTrainedParam('trainingPass', trPass)
        
        if self._useNewModel() and self.skipTraining.get():
          self.skipTraining.set(False)

        preprocessParamsFname= self._getExtraPath("preprocess_params.json")
        preprocParams= self.getPreProcParamsFromForm()
        with open(preprocessParamsFname, "w") as f:
          json.dump(preprocParams, f)

        #Initializing outputs
        self.inSamplingRate = self._getInputMicrographs().getSamplingRate()
        self.USING_INPUT_MICS = False
        self.preCorrectedParSet, self.preCoordSet = [], []

    def preprocessMicsStep(self):
        '''Step which preprocesses the input micrographs'''
        micIds = self.getMicsIds(filterOutNoCoords=False)
        if len(micIds) > 0:
          mics_ = self._getInputMicrographs()
          micsFnameSet = {mics_[micId].getMicName(): mics_[micId].getFileName() for micId in micIds
                          if mics_[micId] is not None}  # to skip failed mics
          self.USING_INPUT_MICS = False
          dirName = os.path.dirname(list(micsFnameSet.values())[0])
          toPreprocessMicFns = self.readyToPreprocessMics(shared=False)
          print('New mics to be preprocessed: %d' % len(toPreprocessMicFns))
          if len(toPreprocessMicFns) > self.PREPROCESS_BATCH_MAX:
              toPreprocessMicFns = toPreprocessMicFns[:self.PREPROCESS_BATCH_MAX]

          if self.ignoreCTF.get():
            preproMicsContent="#mics\n"
            for micFileName in toPreprocessMicFns:
              preproMicsContent+= "%s\n"%os.path.join(dirName, micFileName)
          else:
            preproMicsContent="#mics ctfs\n"
            setOfMicCtf= self.ctfRelations.get()
            assert setOfMicCtf is not None, "Error, CTFs must be provided to compute phase flip"
            for ctf in setOfMicCtf:
              ctf_mic = ctf.getMicrograph()
              ctfFnName = ctf_mic.getFileName()
              if os.path.basename(ctfFnName) in toPreprocessMicFns:
                ctf_mic.setCTF(ctf)
                fnCTF = self._getTmpPath("%s.ctfParam" % os.path.basename(ctfFnName))
                micrographToCTFParam(ctf_mic, fnCTF)
                preproMicsContent+= "%s %s\n"%(ctfFnName, fnCTF)

          inputsFname= self._getTmpPath("preprocMic_inputs.txt")
          ouputDir= self._getExtraPath(self.PRE_PROC_MICs_PATH)
          nThrs= self.numberOfThreads.get()
          with open(inputsFname, "w") as f:
            f.write(preproMicsContent)
          downFactor = self._getDownFactor()
          args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, self.inSamplingRate, downFactor, ouputDir, nThrs)
          if not self.skipInvert.get():
            args+=" --invert_contrast"

          if not self.ignoreCTF.get():
            args+=" --phase_flip"

          self.runJob('xmipp_preprocess_mics', args, numberOfMpi=1)
        self.PREPROCESSING = False

    def insertCaculateConsensusSteps(self, mode, prerequisites):
        '''Insert the steps neccessary for calculating the consensus coordinates of type "mode"'''
        outCoordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE% mode)
        if not os.path.exists(outCoordsDataPath):
          makePath(outCoordsDataPath)
        newDep = self._insertFunctionStep('calculateCoorConsensusStep', outCoordsDataPath, mode, prerequisites=prerequisites)
        newDep = self._insertFunctionStep('loadCoords', outCoordsDataPath, mode, prerequisites=[newDep])
        return [newDep]

    def waitFreeInputMics(self):
      sl = 0
      while self.USING_INPUT_MICS:
        time.sleep(1)
        sl+=1
        print('Waiting mics: ', sl)
        if sl > 60:
            print('WARNING: a thread has been waiting for more than a minute to access the input micrographs')

      self.USING_INPUT_MICS = True

    def waitFreeInputCoords(self):
      sl = 0
      while self.USING_INPUT_COORDS:
        time.sleep(1)
        sl += 1
        print('Waiting coords: ', sl)
        if sl > 60:
          print('WARNING: a thread has been waiting for more than a minute to access the input coordinates')

      self.USING_INPUT_COORDS = True

    def calculateCoorConsensusStep(self, outCoordsDataPath, mode):
        '''Calculates the consensus coordinates from micrographs whose particles haven't been extracted yet in "mode"'''
        #Only calculate consensus for coordinates that has not been extracted yet
        trainedParams = self.loadTrainedParams()
        if trainedParams['trainingPass'] != '' or mode != 'AND':
            if self.checkIfNewMics(mode):
                extractedSetOfCoordsFns = []
                for micFn in self.getExtractedMicFns(mode):
                    extractedSetOfCoordsFns.append(pwutils.path.replaceBaseExt(micFn,'pos'))

                self.waitFreeInputCoords()
                inputCoordsFnames = self.getInpCoordsFns(mode, extractedSetOfCoordsFns)
                self.USING_INPUT_COORDS = False

                inputFileHeader="#pos_i\n"
                inputFileStr=inputFileHeader
                for baseName in inputCoordsFnames:
                    fnames= inputCoordsFnames[baseName]
                    inputFileStr+=" ".join(fnames)+"\n"

                assert len(inputFileStr)>len(inputFileHeader), "Error, no consensus can be computed as there " \
                                                               "is a mismatch in coordinate sets filenames"
                #Concensus decision
                consensus = UNION_INTERSECTIONS if mode=="AND" else OR
                # THE MODE CAN BE A NEW PARAMETER TO BE ASKED TO THE USER
                configFname= self._getTmpPath("consensus_%s_inputs.txt"%(mode) )
                with open(configFname, "w") as f:
                    f.write(inputFileStr)

                args = "-i %s -s %d -c %s -d %f -o %s -t %d" % (configFname, self._getBoxSize(), consensus, self.consensusRadius.get(),
                                                                outCoordsDataPath, self.numberOfThreads.get())
                self.runJob('xmipp_coordinates_consensus', args, numberOfMpi=1)
                self.TO_EXTRACT_MICFNS[mode] = self.readyToExtractMicFns(mode)

                return

    def pickNoise(self):
        '''Find noise coordinates from micrographs in order to use them as negatives in the training process'''
        trainedParams = self.loadTrainedParams()
        if trainedParams['trainingPass'] != '':
            orPosDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % "OR")
            outputPosDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % "NOISE")
            if not "OR" in self.coordinatesDict:  # fill self.coordinatesDict['OR']
              self.loadCoords(orPosDir, 'OR')
            # Getting the extracted mics where noise have not been picked yet
            micsDir = self._getExtraPath(self.PRE_PROC_MICs_PATH)
            toPickNoiseFns = list(set(self.getExtractedMicFns('OR')) -
                                  (set(self.getExtractedMicFns('NOISE')) | set(self.TO_EXTRACT_MICFNS['NOISE'])))
            if len(toPickNoiseFns) > 0:
              micNoiseSet = self.loadMicSetFromFns(micsDir, toPickNoiseFns)
              toNoiseSetOfCoords = self._createSetOfCoordinates(micNoiseSet)
              toNoiseSetOfCoords.setBoxSize(self._getBoxSize())
              readSetOfCoordinates(orPosDir, micSet=micNoiseSet, coordSet=toNoiseSetOfCoords)

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

              argsDict = pickNoise_prepareInput(toNoiseSetOfCoords, self._getTmpPath())
              argsDict['toPickMicsDir'] = toPickMicsDir
              argsDict['toPickCoorsDir'] = toPickCoorsDir
              argsDict["outputPosDir"] = outputPosDir
              argsDict["nThrs"] = self.numberOfThreads.get()
              argsDict["nToPick"] = -1
              args = (" -i %(toPickMicsDir)s -c %(toPickCoorsDir)s -o %(outputPosDir)s -s %(boxSize)s " +
                      "-n %(nToPick)s -t %(nThrs)s") % argsDict

              self.runJob('xmipp_pick_noise', args, numberOfMpi=1)

              self.loadCoords(outputPosDir, 'NOISE', micSet=micNoiseSet)
              self.TO_EXTRACT_MICFNS['NOISE'] = toPickNoiseFns
              print('Adding to extract {} {} micrographs'.format(len(toPickNoiseFns), 'NOISE'))

              return

    def loadCoords(self, posCoorsPath, mode, micSet=[]):
        #Upload coords sqlite
        trainedParams = self.loadTrainedParams()
        if trainedParams['trainingPass'] != '' or mode != 'AND':
            if len(micSet):
                  #Load coordinates from an specific set of mics
                  batchSetOfCoordinates = self._createSetOfCoordinates(micSet)
                  batchSetOfCoordinates.setBoxSize(self._getBoxSize())
                  readSetOfCoordinates(posCoorsPath, micSet=micSet, coordSet = batchSetOfCoordinates)
                  if mode in self.coordinatesDict:
                    for newCoord in batchSetOfCoordinates:
                      apCoord = Coordinate()
                      apCoord.copy(newCoord, copyId=False)
                      self.coordinatesDict[mode].append(apCoord)
                  else:
                       self.coordinatesDict[mode] = batchSetOfCoordinates
            else:
                  sqliteName = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode) + ".sqlite"
                  if os.path.isfile(self._getExtraPath(sqliteName)):
                    cleanPath(self._getExtraPath(sqliteName))

                  self.waitFreeInputCoords()
                  totalSetOfCoordinates = readSetOfCoordsFromPosFnames(posCoorsPath,
                                                                       setOfInputCoords=self.inputCoordinates[0].get(),
                                                                       sqliteOutName=sqliteName, write=True)
                  print("Coordinates %s size: %d" % (mode, totalSetOfCoordinates.getSize()))
                  assert totalSetOfCoordinates.getSize() > MIN_NUM_CONSENSUS_COORDS, \
                    ("Error, the consensus (%s) of your input coordinates was too small (%s). " +
                     "It must be > %s. Try a different input..."
                     ) % (mode, str(totalSetOfCoordinates.getSize()), str(MIN_NUM_CONSENSUS_COORDS))
                  self.coordinatesDict[mode] = totalSetOfCoordinates

            self.USING_INPUT_COORDS = False

    def insertExtractPartSteps(self, mode, prerequisites):
        '''Inserts the steps necessary for extracting the particles from the micrographs'''
        self.newSteps = []
        self.newSteps.append(self._insertFunctionStep("extractParticles", mode, prerequisites= prerequisites))
        self.newSteps.append(self._insertFunctionStep("joinSetOfParticlesStep", mode, prerequisites= self.newSteps))
        return self.newSteps

    def extractParticles(self, mode):
        '''Extract the particles from a set of micrographs with their corresponding coordinates'''
        trainedParams = self.loadTrainedParams()
        if trainedParams['trainingPass'] == '' and (mode.startswith("ADDITIONAL_COORDS") or mode == 'AND' or mode == 'NOISE'):
            print('Training already finished')

        else:
            micsFnameSet = {}
            posDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode)
            preprocMicsPath = self._getExtraPath(self.PRE_PROC_MICs_PATH)
            toExtractMicFns = self.TO_EXTRACT_MICFNS[mode]
            print('To extract {} in mode {}: {}'.format(len(toExtractMicFns), mode, toExtractMicFns))
            if len(toExtractMicFns) <= 0:
              return

            for micFname in toExtractMicFns:
              micFnameBase= pwutils.removeExt(micFname)
              micFname= os.path.join(preprocMicsPath, micFname)
              micsFnameSet[micFnameBase]= micFname
            extractCoordsContent="#mics coords\n"

            if mode.startswith("ADDITIONAL_COORDS"):
              if not os.path.exists(posDir):
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

    def joinSetOfParticlesStep(self, mode, micFns='', trainingPass='', clean=False):
        '''Stores the particles extracted from a set of micrographs in a images.xmd metadata file'''
        trainedParams = self.loadTrainedParams()
        if trainedParams['trainingPass'] == '' and (mode.startswith("ADDITIONAL_COORDS") or mode == 'AND' or mode == 'NOISE'):
            print('Training already finished')

        else:
            fnImages = self._getExtraPath("particles_{}{}.xmd".format(mode, trainingPass))
            posDir = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % mode)
            if micFns == '':
              micFns = self.TO_EXTRACT_MICFNS[mode]
              self.TO_EXTRACT_MICFNS[mode] = []
              print('Updating set of particles written in {}'.format(fnImages))
            else:
              print('Creating set of particles in {}'.format(fnImages))

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
            if not os.path.exists(fnImages) or clean:
              imgsXmd.write(fnImages)
            else:
              imgsXmd.append(fnImages)

        self.EXTRACTING[mode] = False

    def doTraining(self): # PUT SOMETHING LIKE IN THE PCIKNOISE BUT MAKE SURE YOU ARE ONLY USING IT FOR AND SETS
      '''Prepares the positive (AND) and negative (NOISE) coordinates for the training and executes it'''
      if len(self.readyToExtractMicFns('NOISE')) >= self.extractingBatch.get() and not self.EXTRACTING['NOISE']:
        self.EXTRACTING['NOISE'] = True
        depNoise = self._insertFunctionStep('pickNoise', prerequisites=self.initDeps)
        self.newSteps += self.insertExtractPartSteps('NOISE', prerequisites=[depNoise])

      if len(self.readyToExtractMicFns('AND')) >= self.extractingBatch.get() and not self.EXTRACTING['AND']:
        self.EXTRACTING['AND'] = True
        depsAnd = self.insertCaculateConsensusSteps('AND', prerequisites=self.initDeps)
        self.newSteps += self.insertExtractPartSteps('AND', prerequisites=depsAnd)

      trainedParams = self.loadTrainedParams()
      if self.addTrainingData.get() == self.ADD_DATA_TRAIN_CUST and \
              self.trainingDataType == self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS:
        if self.trainTrueSetOfCoords.get() is not None and \
                len(self.readyToExtractMicFns('ADDITIONAL_COORDS_TRUE')) >= self.extractingBatch.get():
          self.TO_EXTRACT_MICFNS['ADDITIONAL_COORDS_TRUE'] = self.readyToExtractMicFns('ADDITIONAL_COORDS_TRUE')
          self.newSteps += self.insertExtractPartSteps('ADDITIONAL_COORDS_TRUE', prerequisites=self.initDeps)
        if self.trainFalseSetOfCoords.get() is not None and \
                len(self.readyToExtractMicFns('ADDITIONAL_COORDS_FALSE')) >= self.extractingBatch.get():
          self.TO_EXTRACT_MICFNS['ADDITIONAL_COORDS_FALSE'] = self.readyToExtractMicFns('ADDITIONAL_COORDS_FALSE')
          self.newSteps += self.insertExtractPartSteps('ADDITIONAL_COORDS_FALSE', prerequisites=self.initDeps)

      if self.cnnFree():
        self.TO_TRAIN_MICFNS = self.readyToTrainMicFns()
        if len(self.TO_TRAIN_MICFNS) >= self.trainingBatch.get():
          self.TRAINING = True
          self.curTrainedParams = trainedParams
          self.depsTrain = [self._insertFunctionStep('trainCNN', self.TO_TRAIN_MICFNS, prerequisites=self.initDeps)]
          self.depsTrain = [self._insertFunctionStep('endTrainingStep', prerequisites=self.depsTrain)]
          self.newSteps += self.depsTrain

    def trainCNN(self, toTrainMicFns):
        '''Trains the CNN with the particles from the ready to train micrographs'''
        trainedParams = self.curTrainedParams
        trPass = trainedParams['trainingPass']
        if not trPass == '':
          trPass += 1
          trainedParams['trainingPass'] = trPass

        #Writting the inputs in xmd
        for mode in ['AND', 'NOISE']:
          self.joinSetOfParticlesStep(mode, toTrainMicFns, trPass, clean=True)
        #Creatting the training pass directory
        netDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass))
        if not os.path.exists(netDataPath):
          makePath(netDataPath)
        nEpochs = self.nEpochs.get()

        #Setting the input and input weights
        posTrainDict = {self._getExtraPath("particles_AND{}.xmd".format(trPass)):  1}
        negTrainDict = {self._getExtraPath("particles_NOISE{}.xmd".format(trPass)):  1}
        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_PRECOMP and trainedParams['firstTraining']:
          negTrainDict[self._getTmpPath("addNegTrainParticles.xmd")]= 1

        if self.usesGpu():
          numberOfThreads = None
          gpuToUse = self.setGPU(oneGPU=True)
        else:
          numberOfThreads = self.numberOfThreads.get()
          gpuToUse = None


        if self.addTrainingData.get() == self.ADD_DATA_TRAIN_CUST:
          if self.trainingDataType.get() == self.ADD_DATA_TRAIN_CUSTOM_OPT_PARTS and trainedParams['firstTraining']:
            if self.trainTrueSetOfParticles.get():
              posTrainFn = self._getExtraPath("trainTrueParticlesSet.xmd")
              posTrainDict[posTrainFn] = self.trainPosWeight.get()
            if self.trainFalseSetOfParticles.get():
              negTrainFn = self._getExtraPath("trainFalseParticlesSet.xmd")
              negTrainDict[negTrainFn] = self.trainNegWeight.get()

          elif self.trainingDataType.get() == self.ADD_DATA_TRAIN_CUSTOM_OPT_COORS:
            if self.trainTrueSetOfCoords.get():
              self.joinSetOfParticlesStep('ADDITIONAL_COORDS_TRUE', toTrainMicFns, trPass, clean=True)
              posTrainFn = self._getExtraPath("particles_ADDITIONAL_COORDS_TRUE{}.xmd".format(trPass))
              posTrainDict[posTrainFn] = self.trainPosWeight.get()
            if self.trainFalseSetOfCoords.get():
              self.joinSetOfParticlesStep('ADDITIONAL_COORDS_FALSE', toTrainMicFns, trPass, clean=True)
              negTrainFn = self._getExtraPath("particles_ADDITIONAL_COORDS_FALSE{}.xmd".format(trPass))
              negTrainDict[negTrainFn] = self.trainNegWeight.get()

        effectiveSize=-1
        nTrueParticles = self.toTrainDataSize.get() if self.toTrainDataSize.get() != -1 else 1e10
        if self._usePretrainedModel():
          if nTrueParticles<1500:
            effectiveSize=1000
          elif 1500<=nTrueParticles<20000:
            effectiveSize=5000
          else:
            effectiveSize=50000
          self.__retrievePreTrainedModel(netDataPath, effectiveSize)
          if self.skipTraining.get():
              nEpochs = 0
        elif not trPass == '' and trPass > 1:
          #Starting from model in previous trainingPass
          self.retrievePreviousPassModel(trPass)

        trainedParams['posParticlesTrained'] += self._getEffectiveNumPartsTrain(posTrainDict)
        fnamesPos, weightsPos= self.__dataDict_toStrs(posTrainDict)
        fnamesNeg, weightsNeg= self.__dataDict_toStrs(negTrainDict)
        args= " -n %s --mode train -p %s -f %s --trueW %s --falseW %s --effective_data_size %s"%(netDataPath,
                      fnamesPos, fnamesNeg, weightsPos, weightsNeg, nTrueParticles)
        args+= " -e %s -l %s -r %s -m %s "%(nEpochs, self.learningRate.get(), self.l2RegStrength.get(),
                                          self.nModels.get())
        if not self.auto_stopping.get():
          args+=" -s"
        if gpuToUse:
          args+= " -g %s"%(gpuToUse)
        if numberOfThreads:
          args+= " -t %s"%(numberOfThreads)

        trainedParams['trainedMicFns'] += self.TO_TRAIN_MICFNS
        trainedParams['firstTraining'] = False
        self.curTrainedParams = trainedParams

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1, env=self.getCondaEnv())
        
    def predictCNN(self):
        '''Predict the particles from the micrographs and calificates the consensus coordinates'''
        trainedParams = self.loadTrainedParams()
        trPass = trainedParams['trainingPass']

        if trPass != '':
          predExten = trPass
          mdORPath = self._getExtraPath("particles_OR.xmd")
        else:
          toPredictMicFns = self.readyToPredictMicFns()
          print("Mics ready to predict {}".format(len(toPredictMicFns)))
          if len(toPredictMicFns) > self.PREDICT_BATCH_MAX:
              toPredictMicFns = toPredictMicFns[:self.PREDICT_BATCH_MAX]

          predExten = '_partial'
          self.joinSetOfParticlesStep(mode='OR', micFns=toPredictMicFns, trainingPass=predExten, clean=True)
          mdORPath = self._getExtraPath("particles_OR{}.xmd".format(predExten))

        netDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass))
        if not os.path.isdir(netDataPath) and self._doContinue():
            prevRunPath = self.continueRun.get()._getExtraPath(self.NET_TEMPLATE.format(trPass))
            copyTree(prevRunPath, netDataPath)
        elif self.skipTraining.get() and self._usePretrainedModel():
          self.__retrievePreTrainedModel(netDataPath)

        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.setGPU(oneGPU=True)
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        mdObject = md.MetaData(mdORPath)
        print('Predicting on {} true particles'.format(mdObject.size()))
              #'in {} micrographs'.format(mdObject.size(), len(toPredictMicFns)))
        predictDict = {mdORPath: 1}

        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get() and not\
                self.loadTrainedParams()['doneExtraTesting']:
            self.uploadTrainedParam('doneExtraTesting', True)
            posTestDict = {self._getExtraPath("testTrueParticlesSet.xmd"): 1}
            negTestDict = {self._getExtraPath("testFalseParticlesSet.xmd"): 1}
        else:
            posTestDict = None
            negTestDict = None

        outParticlesPath = self._getPath(self.PARTICLES_TEMPLATE.format(predExten))
        fnamesPred, weightsPred= self.__dataDict_toStrs(predictDict)

        args= " -n %s --mode score -i %s -o %s "%(netDataPath, fnamesPred, outParticlesPath)

        if posTestDict and negTestDict:
          fnamesPosTest, weightsPosTest= self.__dataDict_toStrs(posTestDict)
          fnamesNegTest, weightsNegTest= self.__dataDict_toStrs(negTestDict)
          args+= " --testingTrue %s --testingFalse %s "%(fnamesPosTest, fnamesNegTest)

        if gpuToUse:
          args+= " -g %s"%(gpuToUse)
        if  numberOfThreads:
          args+= " -t %s"%(numberOfThreads)

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1,
                    env=self.getCondaEnv())

        trainedParams = self.loadTrainedParams()
        if trPass != '':
          trainedParams['predictionPasses'].append(trPass)
        else:
          trainedParams['predictedMicFns'] += toPredictMicFns
        self.saveTrainedParams(trainedParams)

    def createOutputStep(self, closeStream=False):
      trPass = self.loadTrainedParams()['trainingPass']
      if not "OR" in self.coordinatesDict:
        self.loadCoords(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR'), 'OR')

      if trPass == '':
        self.createFinalOutput(closeStream)
      else:
        self.createPreliminarOutput(trPass)

    def createFinalOutput(self, closeStream=False):
      predExten = '_partial'
      partSet = self._createSetOfParticles("outputParts_tmp{}".format(predExten))
      readSetOfParticles(self._getPath(self.PARTICLES_TEMPLATE.format(predExten)), partSet)
      partSet.setSamplingRate(self._getDownFactor() * self.inSamplingRate)

      self.outputParticles, self.outputCoordinates = self.getParticlesOutput(partSet), self.getCoordinatesOutput()
      downFactor = self._getDownFactor()
      for part in partSet:
        coord = part.getCoordinate().clone()
        coord.scale(downFactor)
        deepZscoreLabel = '_xmipp_%s' % emlib.label2Str(md.MDL_ZSCORE_DEEPLEARNING1)
        setattr(coord, deepZscoreLabel, getattr(part, deepZscoreLabel))
        newPart = Particle()
        newPart.copy(part, copyId=False)
        newPart.scaleCoordinate(downFactor)
        if (self.threshold.get() < 0 or
          getattr(newPart, deepZscoreLabel) > self.threshold.get()):
          self.outputCoordinates.append(coord)
          self.outputParticles.append(newPart)

      cleanPattern(self._getPath(self.PARTICLES_TEMPLATE.format(predExten)))
      cleanPattern(self._getPath("*outputParts_tmp{}.sqlite".format(predExten)))
      writeSetOfParticles(self.outputParticles, self._getPath("particles.xmd"))
      self.updateOutput(closeStream)

    def createPreliminarOutput(self, trPass):
      partSet = self._createSetOfParticles("outputParts_tmp{}".format(trPass))
      readSetOfParticles(self._getPath(self.PARTICLES_TEMPLATE.format(trPass)), partSet)
      partSet.setSamplingRate(self._getDownFactor() * self.inSamplingRate)

      self.preliminarOutputParticles = self.getPreParticlesOutput(partSet)
      self.preliminarOutputCoordinates = self.getPreCoordinatesOutput()

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
          self.preliminarOutputCoordinates.append(coord)
          self.preliminarOutputParticles.append(part)

      cleanPattern(self._getPath(self.PARTICLES_TEMPLATE.format(trPass)))
      cleanPattern(self._getPath("*outputParts_tmp{}.sqlite".format(trPass)))
      writeSetOfParticles(self.preliminarOutputParticles, self._getPath(self.PARTICLES_TEMPLATE.format(trPass)))
      self.updatePreOutput(closeStream=True)

    def getPreCoordinatesOutput(self):
      print('Creating new preliminarOutputCoordinates set')
      self.preliminarOutputCoordinates = \
          self._createSetOfCoordinates(self.coordinatesDict['OR'].getMicrographs(asPointer=True))
      self.preliminarOutputCoordinates.copyInfo(self.coordinatesDict['OR'])
      self.preliminarOutputCoordinates.setBoxSize(self._getBoxSize())
      self.preliminarOutputCoordinates.setStreamState(SetOfParticles.STREAM_OPEN)
      self._defineOutputs(preliminarOutputCoordinates=self.preliminarOutputCoordinates)
      self.waitFreeInputCoords()
      for inSetOfCoords in self.inputCoordinates:
        self._defineSourceRelation(inSetOfCoords.get(), self.preliminarOutputCoordinates)
      self.USING_INPUT_COORDS = False
      return self.preliminarOutputCoordinates

    def getCoordinatesOutput(self):
      if not hasattr(self, "outputCoordinates"):
        print('Creating new outputCoordinates set')
        self.outputCoordinates = self._createSetOfCoordinates(self.coordinatesDict['OR'].getMicrographs(asPointer=True))
        self.outputCoordinates.copyInfo(self.coordinatesDict['OR'])
        self.outputCoordinates.setBoxSize(self._getBoxSize())
        self.outputCoordinates.setStreamState(SetOfParticles.STREAM_OPEN)
        self._defineOutputs(outputCoordinates=self.outputCoordinates)
        self.waitFreeInputCoords()
        for inSetOfCoords in self.inputCoordinates:
          self._defineSourceRelation(inSetOfCoords.get(), self.outputCoordinates)
        self.USING_INPUT_COORDS = False
      else:
        # Micrographs of the set removed because there might be new ones in streaming
        self.outputCoordinates.setMicrographs(self.coordinatesDict['OR'].getMicrographs(asPointer=False))
      return self.outputCoordinates

    def getPreParticlesOutput(self, partSet):
      print('Creating new preliminarOutputParticles set')
      self.preliminarOutputParticles = self._createSetOfParticles()
      self.preliminarOutputParticles.copyInfo(partSet)
      self.preliminarOutputParticles.setStreamState(SetOfParticles.STREAM_OPEN)
      self._defineOutputs(preliminarOutputParticles=self.preliminarOutputParticles)
      return self.preliminarOutputParticles

    def getParticlesOutput(self, partSet):
      if not hasattr(self, "outputParticles"):
        print('Creating new outputParticles set')
        self.outputParticles = self._createSetOfParticles()
        self.outputParticles.copyInfo(partSet)
        self.outputParticles.setStreamState(SetOfParticles.STREAM_OPEN)
        self._defineOutputs(outputParticles=self.outputParticles)
      return self.outputParticles

    def updatePreOutput(self, closeStream=False):
      if closeStream:
        self.preliminarOutputCoordinates.setStreamState(SetOfParticles.STREAM_CLOSED)
        self.preliminarOutputParticles.setStreamState(SetOfParticles.STREAM_CLOSED)

      self.preliminarOutputCoordinates.write()
      self.preliminarOutputParticles.write()
      self._store(self.preliminarOutputCoordinates, self.preliminarOutputParticles)

    def updateOutput(self, closeStream=False):
      if closeStream:
        self.outputCoordinates.setStreamState(SetOfParticles.STREAM_CLOSED)
        self.outputParticles.setStreamState(SetOfParticles.STREAM_CLOSED)

      self.outputCoordinates.write()
      self.outputParticles.write()
      self._store(self.outputCoordinates, self.outputParticles)

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

    def retrieveTrainSets(self):
      """ Retrieve, link and return a setOfParticles
          corresponding to the NegativeTrain DeepConsensus trainning set
          with certain extraction conditions (phaseFlip/invContrast)
      """
      prefixYES = ''
      prefixNO = 'no'
      # We always work with inverted contrast particles
      modelType = "negativeTrain_%sPhaseFlip_Invert.mrcs" % (
        prefixNO if self.ignoreCTF.get() else prefixYES)  # mics will be always internally inverted if not done before
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

      cleanPath(self._getPath("particles%s.sqlite" % tmpSqliteSuff))
      return partSet

    def getMicrographFnsWithCoordinates(self, shared=True):
      '''Return a list with the filenames of those microgrpahs which already have coordinates associated in the input
      sets. If shared, it must be in all the sets, if not shared, at least in one'''
      sharedMics = self.getAllCoordsInputMicrographs(shared)
      self.waitFreeInputCoords()
      micPaths = []
      for micFn in sharedMics:
        coordsInMic, mic = [], sharedMics[micFn]
        for coordSet in self.inputCoordinates:
          for coord in coordSet.get().iterCoordinates(mic):
            coordsInMic.append(coord)
            break

        if len(coordsInMic) == len(self.inputCoordinates):
          micPaths.append(mic.getFileName())
      self.USING_INPUT_COORDS = False

      micFns = self.prunePaths(micPaths)
      return micFns

    def getAllCoordsInputMicrographs(self, shared=False):
      '''Returns a dic {micFn: mic} with the input micrographs present associated with all the input coordinates sets.
      If shared, the list contains only those micrographs present in all input coordinates sets, else the list contains
      all microgrpah present in any set (Intersection vs Union)
      Do not create a set, because of concurrency in the database'''
      self.waitFreeInputCoords()
      self.waitFreeInputMics()
      micDict, micFns = {}, set([])
      for inputCoord in self.inputCoordinates:
        newMics = inputCoord.get().getMicrographs()
        newMicFns = []
        for mic in newMics:
          micFn = self.prunePaths([mic.getFileName()])[0]
          micDict[micFn] = mic.clone()
          newMicFns.append(micFn)

        if micFns == set([]) or not shared:
          micFns = micFns | set(newMicFns)
        else:
          micFns = micFns & set(newMicFns)
      self.USING_INPUT_COORDS, self.USING_INPUT_MICS = False, False
      sharedMicDict = {}
      for micFn in micFns:
        sharedMicDict[micFn] = micDict[micFn]

      return sharedMicDict

    def _getInputMicrographs(self):
      '''Return a list with the micrographs corresponding the input coordinates'''
      self.waitFreeInputMics()
      if not hasattr(self, "inputMicrographs") or not self.inputMicrographs:
        self.waitFreeInputCoords()
        if len(self.inputCoordinates) == 0:
          print("WARNING. PROVIDE MICROGRAPHS FIRST")
        else:
          inputMicrographs = self.inputCoordinates[0].get().getMicrographs()
          if inputMicrographs is None:
            raise ValueError("there are problems with your coordiantes, they do not have associated micrographs ")
          self.inputMicrographs = inputMicrographs
        self.USING_INPUT_COORDS = False
      return self.inputMicrographs

    def _getBoxSize(self):
      '''Returns the box size of the input coordinates'''
      if not hasattr(self, "boxSize") or not self.boxSize:
        self.waitFreeInputCoords()
        firstCoords = self.inputCoordinates[0].get()
        self.USING_INPUT_COORDS = False
        self.boxSize = firstCoords.getBoxSize()
        self.downFactor = self.boxSize / float(DEEP_PARTICLE_SIZE)
      return self.boxSize

    def _getDownFactor(self):
      if not hasattr(self, "downFactor") or not self.downFactor:
        self.boxSize = self._getBoxSize()
        self.downFactor = self.boxSize / float(DEEP_PARTICLE_SIZE)
        assert self.downFactor >= 1, \
          "Error, the particle box size must be greater or equal than 128."

      return self.downFactor

    def _getConsensusParticlesDir(self, mode):
      pathFun = self._getTmpPath if mode != "OR" else self._getExtraPath
      return pathFun(self.CONSENSUS_PARTS_PATH_TEMPLATE% mode)

    def loadMicSetFromFns(self, inputDir, micFns):
      '''Returns a set of Micrographs from their filenames'''
      micSet = self._createSetOfMicrographs()
      for micFn in micFns:
        micPath = os.path.join(inputDir, micFn)
        micSet.append(Micrograph(micPath))
      micSet.copyInfo(self._getInputMicrographs())
      self.USING_INPUT_MICS = False
      return micSet

    def getPreProcParamsFromForm(self):
        mics_= self._getInputMicrographs()
        fnMic = mics_.getFirstItem().getFileName()
        self.USING_INPUT_MICS = False
        pathToMics= os.path.split(fnMic)[0]
        pathToCtfs= "None"

        if not self.ignoreCTF.get():
          pathToCtfs=  os.path.split(self.ctfRelations.get().getFileName())[0]

        paramsInfo={"mics_ignoreCTF": self.ignoreCTF.get(),
                    "mics_skipInvert": self.skipInvert.get(),
                    "mics_pathToMics": pathToMics,
                    "mics_pathToCtfs": pathToCtfs}
        self.waitFreeInputCoords()
        coordsNames=[]
        for inputCoords in self.inputCoordinates:
          coordsNames.append( inputCoords.get().getFileName() )
        self.USING_INPUT_COORDS = False
        coordsNames= tuple(sorted(coordsNames))
        paramsInfo["coords_pathCoords"]= coordsNames

        return paramsInfo

    def __dataDict_toStrs(self, dataDict):
        fnamesStr=[]
        weightsStr=[]
        for fname in dataDict:
          fnamesStr.append(fname)
          weightsStr.append(str(dataDict[fname]) )
        return ":".join(fnamesStr), ":".join(weightsStr)

    def _getEffectiveNumPartsTrain(self, dictTrueData):
        '''Returns the number of particles being used for training'''
        nParts=0
        for mdPath in dictTrueData:
          mdObject = md.MetaData(mdPath)
          nParts+= mdObject.size()
        return nParts

    def loadMeanAccuracy(self):
        trainedParams = self.curTrainedParams
        trPass = trainedParams['trainingPass']
        netDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass))
        netMeanAccFname = os.path.join(netDataPath, "netsMeanValAcc.txt")
        if os.path.exists(netDataPath):
            with open(netMeanAccFname) as f:
                lines = f.readlines()
                mean_accuracy = float(lines[0].split()[1])
            return mean_accuracy
        else:
            return None

    #STREAMING and state checks
    def trainingOn(self):
      '''Return a boolean for whether to perform training. True if the training must not be skipped and if the finish
      training criteria has not been reached yet'''
      trainedParams = self.loadTrainedParams()
      return not self.skipTraining.get() and trainedParams['keepTraining']

    def predictionsOn(self):
      '''Return a boolean for whether to perform a prediction. True if there must be a preliminar prediction or
      if the training process has finished (trainingPass=='') '''
      trainedParams = self.loadTrainedParams()
      return (self.readyPreliminarPrediction() or trainedParams['trainingPass'] == '')

    def readyPreliminarPrediction(self):
      '''Return a boolean for whether to perform a preliminar predition. True if the user set it and the current
      trained network has not been used yet'''
      if self.networkReadyToPredict():
        trainedParams = self.loadTrainedParams()
        if not trainedParams['trainingPass'] in trainedParams[
          'predictionPasses'] and self.doPreliminarPredictions.get():
          return True
      return False

    def networkReadyToPredict(self):
      '''Returns true if the CNN is trained or the user specified it does not need to be trained'''
      trainedParams = self.loadTrainedParams()
      return (self.skipTraining.get() and len(self.readyToPredictMicFns()) > 0) \
             or len(trainedParams['trainedMicFns']) > 0

    def cnnFree(self):
      return not self.PREDICTING and not self.TRAINING

    def allFree(self):
      '''Kind of "traficlight" that specifies if there is not extraction, training or prediction going on, which would
      alterate the states of the protocol'''
      gExtracting = False
      for mode in ['OR', 'NOISE', 'AND']:
        if self.EXTRACTING[mode]:
          gExtracting = True

      return not self.PREDICTING and not self.TRAINING and not gExtracting and not self.PREPROCESSING

    def checkIfParentsFinished(self):
      '''Check the streamState of the coordinates input to check if the parent protocols are finsihed'''
      self.waitFreeInputCoords()
      finished=True
      for coords in self.inputCoordinates:
        coords = coords.get()
        coords.loadAllProperties()
        if coords.isStreamOpen():
          finished = False
          break
      self.USING_INPUT_COORDS = False
      return finished

    def checkIfNewMics(self, mode=''):
      '''Check if the are new micrographs ready for extracting particles'''
      if mode == '':
        for mode in ['OR', 'NOISE', 'AND']:
          if len(self.readyToExtractMicFns(mode)) > 0:
            return True
      else:
        if len(self.readyToExtractMicFns(mode)) > 0:
          return True
      return False

    #Get data attributes
    def getMicsIds(self, filterOutNoCoords=False):
        '''Returns the input micrographs Ids'''
        if not filterOutNoCoords:
          idSet = self._getInputMicrographs().getIdSet()
          self.USING_INPUT_MICS = False
          return idSet
        self.waitFreeInputCoords()
        micFnames, micIds = set([]), set([])
        for coordinatesP in self.inputCoordinates:
            for coord in coordinatesP.get():
              micIds.add( coord.getMicId())
              micFnames.add( coord.getMicName() )
        self.USING_INPUT_COORDS = False
        return sorted( micIds )

    def getInputMicsFns(self, shared):
      '''Returns the input micrographs filenames'''
      return sorted(self.getAllCoordsInputMicrographs(shared).keys())

    def prunePaths(self, paths):
      fns = []
      for path in paths:
        fns.append(path.split('/')[-1])
      return fns

    def getPreprocessedMicFns(self):
      '''Return the list of preprocessed micrograph filenames'''
      prepDir = self._getExtraPath(self.PRE_PROC_MICs_PATH)
      if not os.path.exists(prepDir):
        return []
      return os.listdir(prepDir)

    def getExtractedMicFns(self, mode):
      '''Return the list of extracted micrograph filenames
      (micrographs where particles of type "mode" have been extracted)'''
      outputDir = self._getConsensusParticlesDir(mode)
      if not os.path.exists(outputDir):
        return []
      fns, micFns = os.listdir(outputDir), []
      for fn in fns:
        micFns.append(pwutils.path.replaceBaseExt(fn, 'mrc'))
      return list(set(micFns))

    def getTrainedMicFns(self):
      '''Return the list of microgrpahs whose particles have been used for training'''
      trainedParams = self.loadTrainedParams()
      return trainedParams['trainedMicFns']

    def getPredictedMicFns(self):
      '''Return the list of microgrpahs whose particles have been used for prediction'''
      trainedParams = self.loadTrainedParams()
      return trainedParams['predictedMicFns']

    #Get ready sets
    def readyToPreprocessMics(self, shared):
      '''Return the list of micrograph filenames which are ready to be preprocessed and have not been preprocessed yet'''
      micFns = self.getInputMicsFns(shared)
      return list(set(micFns) - set(self.getPreprocessedMicFns()))

    def readyToExtractMicFns(self, mode):
      '''Return the list of micrograph filenames which are ready to be extracted (preprocessed and common for all inputs)
      and have not or are not being extracted yet'''
      return list((set(self.getPreprocessedMicFns()) & set(self.getMicrographFnsWithCoordinates(shared=True))) -
                  (set(self.getExtractedMicFns(mode)) | set(self.TO_EXTRACT_MICFNS[mode])))

    def readyToTrainMicFns(self):
      '''Return the list of micrograph filenames which are ready to be used for training and
      have not or are not being trained yet'''
      extractedMicFns = set(self.getExtractedMicFns('OR')) & set(self.getExtractedMicFns('NOISE')) & \
                        set(self.getExtractedMicFns('AND'))
      readyToTrain = list(extractedMicFns - set(self.getTrainedMicFns()))
      return readyToTrain[:min(len(readyToTrain), self.TRAIN_BATCH_MAX)]

    def readyToPredictMicFns(self):
      '''Return the list of micrograph filenames which are ready to be used for prediction and
      have not or are not being predicted yet'''
      extractedMicFns = set(self.getExtractedMicFns('OR'))
      readyToPredict = list(extractedMicFns - set(self.getPredictedMicFns()))
      return readyToPredict

    def getInpCoordsFns(self, mode, extractedSetOfCoordsFns):
      Tm = []
      for coordinatesP in self.inputCoordinates:
        mics = coordinatesP.get().getMicrographs()
        Tm.append(mics.getSamplingRate())
      nCoordsSets = len(Tm)

      inputCoordsFnames = {}
      for coord_num, coordinatesP in enumerate(self.inputCoordinates):
        tmpPosDir = self._getTmpPath("input_coords_%d_%s" % (coord_num, mode))
        if not os.path.exists(tmpPosDir):
          makePath(tmpPosDir)
        writeSetOfCoordinates(tmpPosDir, coordinatesP.get(), scale=float(Tm[coord_num]) / float(Tm[0]))
        for posFname in os.listdir(tmpPosDir):
          baseName, extension = os.path.splitext(os.path.basename(posFname))
          if extension == ".pos" and not posFname in extractedSetOfCoordsFns:
            if baseName not in inputCoordsFnames:
              inputCoordsFnames[baseName] = ["None"] * nCoordsSets
            inputCoordsFnames[baseName][coord_num] = os.path.join(tmpPosDir, posFname)
      return inputCoordsFnames

    #Training params utils
    def loadTrainedParams(self):
      '''Load the dictionary stored in pickle format which stores the trained parameters.
      Creates a initial one if it does not exist yet'''
      paramsFile = self._getExtraPath(self.TRAINED_PARAMS_PATH)
      if os.path.exists(paramsFile):
        with open(paramsFile, 'rb') as handle:
          params = pickle.load(handle)
      else:
        params = {'trainedMicFns': [],
                  'predictedMicFns': [],
                  'posParticlesTrained': 0,
                  'trainingPass': 0,
                  'predictionPasses': [],
                  'doneExtraTesting': False,
                  'firstTraining': True,
                  'keepTraining': True,
                  }
      return params

    def uploadTrainedParam(self, keyParam, newValue):
      '''Upload the value of a parameter from the trained parameters'''
      params = self.loadTrainedParams()
      params[keyParam] = newValue
      self.saveTrainedParams(params)

    def saveTrainedParams(self, params):
      '''Save the trained parameters dictionary'''
      paramsFile = self._getExtraPath(self.TRAINED_PARAMS_PATH)
      with open(paramsFile, 'wb') as handle:
        pickle.dump(params, handle)

    #CNN models utils
    def retrievePreviousPassModel(self, trPass, lastTrPass=''):
      '''Retrieves a previous CNN model and copies its folders to used the network in a new location'''
      curNetDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass))
      if trPass == '':
        prevNetDataPath = self._getExtraPath(self.NET_TEMPLATE.format(lastTrPass))
      else:
        prevNetDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass - 1))
      if prevNetDataPath != curNetDataPath:
        copyTree(prevNetDataPath, curNetDataPath)

    def retrievePreviousRunModel(self, prevProt, trPass=''):
      '''Retrieves a CNN model from other protocol and copies its folders to used the network in a new location'''
      curNetDataPath = self._getExtraPath(self.NET_TEMPLATE.format(trPass))
      prevNetDataPath = prevProt._getExtraPath("nnetData")
      if prevNetDataPath != curNetDataPath:
        copyTree(prevNetDataPath, curNetDataPath)

    def __retrievePreTrainedModel(self, netDataPath, effectiveSize=-1):
        '''Retrieves a previously trained CNN model'''
        if effectiveSize==-1:
          effectiveSize=int(5e4)
        modelTypeDir= "keras_models/%sPhaseFlip_Invert/nnetData_%d/tfchkpoints_0" % (
                            "no" if self.ignoreCTF.get() else "", effectiveSize)
        modelTypeDir= self.getModel("deepConsensus", modelTypeDir)

        for i in range(self.nModels.get()):
          targetPath= os.path.join(netDataPath, "tfchkpoints_%d"%(i))
          print(targetPath, modelTypeDir)
          copyTree(modelTypeDir, targetPath)


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
