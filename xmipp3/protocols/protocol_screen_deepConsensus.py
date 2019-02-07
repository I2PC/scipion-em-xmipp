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
from math import sqrt
from glob import glob
import numpy as np
import json
from pyworkflow.utils.path import makePath, cleanPattern, cleanPath, copyTree, createLink, copyFile
from pyworkflow.protocol.constants import *
from pyworkflow.em.constants import RELATION_CTF, ALIGN_NONE
from pyworkflow.em.convert import ImageHandler
from pyworkflow.em.data import SetOfCoordinates, Coordinate, SetOfParticles
from pyworkflow.em.protocol import ProtParticlePicking

import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.em.metadata as MD

import xmippLib as xmipp

import xmipp3
from xmipp3 import XmippProtocol
from xmipp3.protocols.protocol_pick_noise import pickNoise_prepareInput
from xmipp3.protocols.coordinates_tools.io_coordinates import readSetOfCoordsFromPosFnames, writeCoordsListToPosFname
from xmipp3.convert import readSetOfParticles, setXmippAttributes, micrographToCTFParam, writeSetOfParticles

DEEP_PARTICLE_SIZE = 128
MIN_NUM_CONSENSUS_COORDS = 256

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
    CONSENSUS_COOR_PATH_TEMPLATE="consensus_%s"

    ADD_TRAIN_TYPES = ["None", "Precompiled", "Custom"]
    ADD_TRAIN_NONE = 0
    ADD_TRAIN_MODEL = 1
    ADD_TRAIN_CUST = 2

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_SERIAL

    def _defineParams(self, form):
        # GPU settings
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Use GPU (vs CPU)",
                       help="Set to true if you want to use GPU implementation "
                            "of Optical Flow.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addSection(label='Input')
        #CONTINUE FROM PREVIOUS TRAIN
        form.addParam('doContinue', params.BooleanParam, default=False,
                      label='Use previously trained model?',
                      help='If you set to *Yes*, you should select a previous '
                           'run of type *%s* or *%s* class and some of '
                           'the input parameters will be taken from it.'
                           % (self.getClassName(), None))
        form.addParam('continueRun', params.PointerParam,
                      pointerClass=self.getClassName(),
                      condition='doContinue', allowsNull=True,
                      label='Select previous run',
                      help='Select a previous run to continue from.')
        form.addParam('keepTraining', params.BooleanParam,
                      default=True, condition='doContinue',
                      label='Continue training on previously trained model?',
                      help='If you set to *Yes*, you should provide training set.')

        # CONTINUE PARAMETERS
        form.addParam('inputCoordinates', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
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
                           '2) mic normalization to 0 mean and 1 std and [OPTIONALLY, mic contrast inversion].\n   E.g. '
                           ' xmipp_transform_normalize -i out1/101_movie_aligned.mrc -o out2/101_movie_aligned.mrc --method '
                           'OldXmipp [ --invert ]\n'
                           '3) particles extraction.\n   E.g. xmipp_micrograph_scissor  -i out2/101_movie_aligned.mrc '
                           '--pos particles@Runs/101_movie_aligned.pos -o out3/105_movie_aligned_particles '
                           ' --Xdim 128 --downsampling newSamplingRate --fillBorders  ( Correct your coordinates with '
                           'newSamplingRate if needed)\n'
                           '4) OPTIONAL: phase flipping using CTF.\n xmipp_ctf_phase_flip  -i '
                           'particles/105_movie_aligned_noDust.xmp -o particles/105_movie_aligned_flipped.xmp '
                           '--ctf ctfPath/105_movie_aligned.ctfParam --sampling newSamplingRate')

        form.addParam('doInvert', params.BooleanParam, default=False,
                      label='Invert contrast', 
                      help='If you invert the contrast, your particles will be white over a black background in the micrograph. '
                           'We prefer white particles. Select *Yes* if you have not inverted the constrast in the micrograph'
                           ' so that we can extract white particles')
        
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
                           'Make it bigger if suffering overfitting (validation acc decreases but training acc increases)\n'
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

        form.addSection(label='Additional training data')
        form.addParam('addTrainingData', params.EnumParam,
                      choices=self.ADD_TRAIN_TYPES,
                      default=self.ADD_TRAIN_NONE,
                      label='Additional training data',
                      help='If you set to *%s*, you should select positive and/or '
                           'negative sets of particles. Regard that our method, '
                           'internally, uses particles that are extracted from '
                           'preprocess micrographs. Steps are:\n'
                           '1) mic donwsampling to the required size such that '
                           'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                           ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                           '2) mic normalization to 0 mean and 1 std and [OPTIONALLY, mic contrast inversion].\n   E.g. '
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
                           % (self.ADD_TRAIN_CUST, self.ADD_TRAIN_MODEL,
                              self.ADD_TRAIN_NONE))
        form.addParam('trainPosSetOfParticles', params.PointerParam,
                      label="Positive train particles 128px (optional)",
                      pointerClass='SetOfParticles', allowsNull=True,
                      condition='addTrainingData==%s'%self.ADD_TRAIN_CUST,
                      help='Select a set of true positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainPosWeight', params.IntParam, default='1',
                      label="Weight of positive additional train particles",
                      condition='addTrainingData==%s' % self.ADD_TRAIN_CUST,
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'positive particles.The weight value indicates '
                           'internal particles are weighted with 1. '
                           'If weight is -1, weight will be calculated such that '
                           'the contribution of additional data is equal to '
                           'the contribution of internal particles')
        form.addParam('trainNegSetOfParticles', params.PointerParam,
                      label="Negative train particles 128px (optional)",
                      pointerClass='SetOfParticles',  allowsNull=True,
                      condition='addTrainingData==%s'%self.ADD_TRAIN_CUST,
                      help='Select a set of false positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainNegWeight', params.IntParam, default='1',
                      label="Weight of negative additional train particles",
                      condition='addTrainingData==%s' % self.ADD_TRAIN_CUST,
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'negative particles. The weight value indicates '
                           'the number of times each image may be included at '
                           'most per epoch. Deep consensus internal particles '
                           'are weighted with 1. If weight is -1, weight '
                           'will be calculated such that the contribution of '
                           'additional data is equal to the contribution of '
                           'internal particles')

        form.addParallelSection(threads=2, mpi=0)

    def _validate(self):
        errorMsg = []
        if self._getBoxSize()< DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, too small particles (needed 128 px), "
                          "have you provided already downsampled micrographs? "
                          "If so, use original ones")
        if not self.ignoreCTF.get() and self.ctfRelations.get() is None:
          errorMsg.append("Error, CTFs must be provided to compute phase flip. "
                          "Please, provide a set of CTFs.")
        return errorMsg

#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):

        self.inputMicrographs = None
        self.boxSize = None
        self.coordinatesDict = {}

        allMicIds = self._getInputMicrographs().getIdSet()

        deps = []
        deps.append(self._insertFunctionStep("initializeStep", prerequisites=deps))
        depsPrepMic = self.insertMicsPreprocSteps(allMicIds, prerequisites=deps)
        # OR before noise always
        depsOr_cons = self.insertCaculateConsensusSteps(allMicIds, 'OR',
                                                        prerequisites=depsPrepMic)
        depsOr = self.insertExtractPartSteps(allMicIds, 'OR',
                                             prerequisites=depsOr_cons)
        depsTrain = depsOr
        if not self.doContinue.get() or self.keepTraining.get():
            depNoise = self._insertFunctionStep('pickNoise', prerequisites=depsOr)
            depsNoise = self.insertExtractPartSteps(allMicIds, 'NOISE',
                                                    prerequisites=[depNoise])

            depsAnd = self.insertCaculateConsensusSteps(allMicIds, 'AND',
                                                        prerequisites=depsOr)
            depsAnd = self.insertExtractPartSteps(allMicIds, 'AND',
                                                  prerequisites=depsAnd)

            depsTrain = depsTrain + depsNoise + depsAnd

        if not self.doContinue.get() or self.keepTraining.get():
          depTrain = self._insertFunctionStep('trainCNN', prerequisites=depsTrain)
        else:
          depTrain=None
        depPredict = self._insertFunctionStep('predictCNN', prerequisites= [depTrain] if not depTrain is None else [])
        self._insertFunctionStep('createOutputStep', prerequisites=[depPredict])

    def initializeStep(self):
        """
            Create paths where data will be saved
        """
                            
        makePath(self._getExtraPath('preProcMics'))
        if self.doTesting.get() and self.testPosSetOfParticles.get() and self.testNegSetOfParticles.get():
            writeSetOfParticles(self.testPosSetOfParticles.get(),
                                self._getExtraPath("testTrueParticlesSet.xmd"))
            writeSetOfParticles(self.testNegSetOfParticles.get(),
                                self._getExtraPath("testFalseParticlesSet.xmd"))

        if self.addTrainingData.get() == self.ADD_TRAIN_CUST:
            if self.trainPosSetOfParticles.get():
                writeSetOfParticles(self.trainPosSetOfParticles.get(),
                                    self._getExtraPath("trainTrueParticlesSet.xmd"))
            if self.trainNegSetOfParticles.get():
                writeSetOfParticles(self.trainNegSetOfParticles.get(),
                                    self._getExtraPath("trainFalseParticlesSet.xmd"))
        elif self.addTrainingData.get() == self.ADD_TRAIN_MODEL:
            writeSetOfParticles(self.retrieveTrainSets(),
                                self._getTmpPath("addNegTrainParticles.xmd"))

        if self.checkIfPrevRunIsCompatible():
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
        modelType = "negativeTrain_%sPhaseFlip_%sInvert.mrcs" % (
                    prefixYES if self.doInvert.get() else prefixNO,
                    prefixYES if self.ignoreCTF.get() else prefixNO)
        modelPath = xmipp3.Plugin.getModel("deepConsensus", modelType)
        print("Precompiled negative particles found at %s"%( modelPath))
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
              self.inputMicrographs = self.inputCoordinates[0].get().getMicrographs()
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


    def insertCaculateConsensusSteps(self, micIds, mode, prerequisites):
        #TODO: make it parallel. It does not work due to concurrency and sqlite cursor
        
        deps = []

        consensus = -1 if mode=="AND" else 1
        coordsDataPath = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE% mode)
        if not self.checkIfPrevRunIsCompatible( "coords_"):
          dep_ = prerequisites
          makePath(self._getExtraPath(coordsDataPath))
          for micId in micIds:
            dep = self._insertFunctionStep('calculateConsensusStep', micId,
                                       coordsDataPath, consensus, prerequisites=dep_)
            dep_ = [dep]
            deps.append(dep)
          newDep = self._insertFunctionStep('loadCoords', coordsDataPath,
                                          mode, True, prerequisites=deps)
          deps = [newDep]
        else:
          deps= prerequisites
        return deps
        
    def calculateConsensusStep(self, micId, coordsDataPath, consensus):

        Tm = []
        mic_fname=None
        for coordinates in self.inputCoordinates:
            mics= coordinates.get().getMicrographs()
            Tm.append(mics.getSamplingRate())
            if mic_fname==None:
              try:
                mic_fname= mics[micId].getFileName()
              except KeyError as e:
                print("Exception",e)
                raise e
        
        # Get all coordinates for this micrograph
        coords = []
        Ncoords = 0
        n = 0

        for coordinates in self.inputCoordinates:
            coordArray = np.asarray([x.getPosition() for x in
                                     coordinates.get().iterCoordinates(micId)],
                                    dtype=float)
            coordArray *= float(Tm[n])/float(Tm[0])
            coords.append(np.asarray(coordArray, dtype=int))
            Ncoords += coordArray.shape[0]
            n += 1
        
        allCoords = np.zeros([Ncoords, 2])
        votes = np.zeros(Ncoords)
        
        # Add all coordinates in the first method
        N0 = coords[0].shape[0]
        inAllMicrographs = consensus <= 0 or consensus == len(self.inputCoordinates)
        if N0 == 0 and inAllMicrographs:
            return
        elif N0 > 0:
            allCoords[0:N0, :] = coords[0]
            votes[0:N0] = 1
        
        boxSize = self._getBoxSize()
        consensusNpixels = self.consensusRadius.get() * boxSize

        # Add the rest of coordinates
        Ncurrent = N0
        for n in range(1, len(self.inputCoordinates)):
            for coord in coords[n]:
                if Ncurrent > 0:
                    dist = np.sum((coord - allCoords[0:Ncurrent])**2, axis=1)
                    imin = np.argmin(dist)
                    if sqrt(dist[imin]) < consensusNpixels:
                        newCoord = (votes[imin]*allCoords[imin,]+coord)/(votes[imin]+1)
                        allCoords[imin,] = newCoord
                        votes[imin] += 1
                    else:
                        allCoords[Ncurrent, :] = coord
                        votes[Ncurrent] = 1
                        Ncurrent += 1
                else:
                    allCoords[Ncurrent, :] = coord
                    votes[Ncurrent] = 1
                    Ncurrent += 1

        # Select those in the consensus
        if consensus <= 0:
            consensus = len(self.inputCoordinates)

        consensusCoords = allCoords[votes >= consensus, :]
        # Write the consensus file only if there are some coordinates (size > 0)
        if consensusCoords.size>0:
            writeCoordsListToPosFname(mic_fname, consensusCoords, outputRoot=coordsDataPath)

    def loadCoords(self, posCoorsPath, mode, writeSet=False):
        boxSize = self._getBoxSize()
        inputMics = self._getInputMicrographs()
        sqliteName= self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode)+".sqlite"
        if os.path.isfile(self._getExtraPath(sqliteName)):
            cleanPath(self._getExtraPath(sqliteName))
        print(posCoorsPath, mode, sqliteName)
        setOfCoordinates= readSetOfCoordsFromPosFnames(posCoorsPath, setOfInputCoords= self.inputCoordinates[0].get(),
                                   sqliteOutName= sqliteName, write=writeSet )
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
                               'OR', writeSet=False)

          argsDict=pickNoise_prepareInput(self.coordinatesDict['OR'], self._getTmpPath())
          argsDict["outputPosDir"]= outputPosDir
          argsDict["nThrs"] = self.numberOfThreads.get()
          argsDict["nToPick"]=-1
          args=(" -i %(mics_dir)s -c %(inCoordsPosDir)s -o %(outputPosDir)s -s %(boxSize)s "+
                "-n %(nToPick)s -t %(nThrs)s")%argsDict
          
          if not self.checkIfPrevRunIsCompatible( "coords_"):                
              self.runJob('xmipp_pick_noise', args, numberOfMpi=1)
              
          self.loadCoords(outputPosDir, 'NOISE', writeSet=True)
          
        
    def getPreProcParamsFromForm(self):
        mics_= self._getInputMicrographs()
        mic = mics_.getFirstItem()
        fnMic = mic.getFileName()
        pathToMics= os.path.split(fnMic)[0]
        pathToCtfs= "None"
        
        if not self.ignoreCTF.get():
          pathToCtfs=  os.path.split(self.ctfRelations.get().getFileName())[0]

        paramsInfo={"mics_ignoreCTF":self.ignoreCTF.get(), "mics_doInvert":self.doInvert.get(), 
                    "mics_pathToMics":pathToMics, "mics_pathToCtfs": pathToCtfs}
        coordsNames=[]            
        for inputCoords in self.inputCoordinates:
          coordsNames.append( inputCoords.get().getFileName() )
        coordsNames= tuple(sorted(coordsNames))
        paramsInfo["coords_coordsPath"]= coordsNames

        return paramsInfo

    def checkIfPrevRunIsCompatible(self, inputType=""):
        '''
        inputType can be mics_ or coords_
        '''
        def _makeTupleIfList(candidateList):
          if isinstance(candidateList, list):
            return tuple(candidateList)
          else:
            return candidateList
        preprocParams= self.getPreProcParamsFromForm()
        preprocParams= { k:preprocParams[k] for k in preprocParams if k.startswith(inputType) }
        if self.doContinue.get():
            preprocessParamsFname = self.continueRun.get()._getExtraPath("preprocess_params.json")
            with open(preprocessParamsFname) as f:
              preprocParams_loaded = json.load(f)
            preprocParams_loaded= { k:_makeTupleIfList(preprocParams_loaded[k]) for k in preprocParams_loaded 
                                            if k.startswith(inputType) }          
            shared_items = {k: preprocParams_loaded[k] for k in preprocParams_loaded if 
                                  k in preprocParams and preprocParams_loaded[k] == preprocParams[k]}
                                  
            return len(shared_items)==len(preprocParams) and len(shared_items)==len(preprocParams_loaded)
        return False
        
    def insertMicsPreprocSteps(self, micIds, prerequisites):
        deps = []
        boxSize = self._getBoxSize()
        self.downFactor = boxSize / float(DEEP_PARTICLE_SIZE)
        mics_ = self._getInputMicrographs()
        samplingRate = self._getInputMicrographs().getSamplingRate()

        preproDep = self._insertFunctionStep("preprocessMicsInitStep", micIds, prerequisites=prerequisites)
                                             
        if self.checkIfPrevRunIsCompatible( "mics_"):
            print("copying preprocess mics")
            createLink( self.continueRun.get()._getExtraPath('preProcMics'), 
                      self._getExtraPath('preProcMics'))
            alreadyPreProMics= set([  fname for fname in os.listdir(self._getExtraPath('preProcMics'))])
        else:
          alreadyPreProMics= []
        for micId in micIds:
            mic = mics_[micId]
            fnMic = mic.getFileName()
            if not os.path.basename(fnMic) in alreadyPreProMics:
              deps.append(self._insertFunctionStep("preprocessOneMicStep",
                                                   micId, fnMic, samplingRate,
                                                   prerequisites=[preproDep]))

        return deps


    def preprocessMicsInitStep(self, micIds):
        self._getDownFactor()
        mics_ = self._getInputMicrographs()

        if not self.ignoreCTF.get():
            
            setOfMicCtf= self.ctfRelations.get()
            if setOfMicCtf.getSize() != len(micIds):
              raise ValueError("Error, there are different number of CTFs compared to "+
                                "the number of micrographs where particles were picked")
            else:
                assert setOfMicCtf is not None, \
                        "Error, CTFs must be provided to compute phase flip"

                self.micToCtf = {}
                micsFnameSet = set( [ mic.getMicName() for mic in mics_])

                for ctf in setOfMicCtf:
                    ctf_mic = ctf.getMicrograph()
                    ctfMicName = ctf_mic.getMicName()

                    if ctfMicName in micsFnameSet:
                        self.micToCtf[ctfMicName] = ctf_mic
                        ctf_mic.setCTF(ctf)        

    def preprocessOneMicStep(self, micId, fnMic, samplingRate):
        downFactor = self._getDownFactor()
        fnPreproc = self._getExtraPath('preProcMics', os.path.basename(fnMic))

        if downFactor != 1: 
            args = "-i %s -o %s --step %f --method fourier" % (fnMic, fnPreproc, downFactor)
            self.runJob('xmipp_transform_downsample', args, numberOfMpi=1)
            fnMic = fnPreproc
            
        if not self.ignoreCTF.get():
            if not hasattr(self, "micToCtf"):
                self.preprocessMicsInitStep()

            fnCTF = self._getTmpPath("%s.ctfParam" % os.path.basename(fnMic))
            micrographToCTFParam(self.micToCtf[os.path.basename(fnMic)], fnCTF)
            
            args = " -i %s -o %s --ctf %s --sampling %f"
            self.runJob('xmipp_ctf_phase_flip',
                        args % (fnMic, fnPreproc, fnCTF, samplingRate*downFactor),
                        numberOfMpi=1)
            fnMic = fnPreproc
            
        args = "-i %s -o %s --method OldXmipp" % (fnMic, fnPreproc)
        if self.doInvert.get():
            args += " --invert"
        self.runJob('xmipp_transform_normalize', args, numberOfMpi=1)
        
    def insertExtractPartSteps(self, micIds, mode, prerequisites):
        deps = []
        mics_ = self._getInputMicrographs()
        if not self.checkIfPrevRunIsCompatible(""):
            newDep = self._insertFunctionStep("prepareExtractParticles", mode,
                                                prerequisites= prerequisites)
            prerequisites = [newDep]
            for micId in micIds:
                mic = mics_[micId]
                fnMic = mic.getFileName()
                fnMic = self._getExtraPath('preProcMics', os.path.basename(fnMic))
##                fnPos = self._getExtraPath('coord_%s' % mode,
                fnPos = self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE% mode,
                                           pwutils.replaceBaseExt(fnMic, "pos"))
                deps.append(self._insertFunctionStep("extractParticlesStep", fnMic,
                                                     fnPos, mode, numberOfMpi=1,
                                                     prerequisites= prerequisites))

            newDep= self._insertFunctionStep("joinSetOfParticlesStep", micIds, mode,
                                             prerequisites= deps)
            deps = [newDep]
        else:
            deps = prerequisites

        return deps


    def prepareExtractParticles(self, mode):
        self.loadCoords(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE %(mode)),
                        mode, writeSet=True)
        makePath(self._getExtraPath('parts_%s' % mode))


    def extractParticlesStep(self, fnMic, fnPos, mode):
        if os.path.isfile(fnPos):
            fnPos = "particles@"+fnPos            
            outputStack = self._getExtraPath('parts_%s' % mode,
                                             pwutils.removeBaseExt(fnMic))
            args = " -i %s --pos %s" % (fnMic, fnPos)
            args += " -o %s --Xdim %d" % (outputStack, int( self._getBoxSize() / self._getDownFactor()))
            args += " --downsampling %f --fillBorders" % self._getDownFactor()
            self.runJob("xmipp_micrograph_scissor", args, numberOfMpi=1)
        else:
            print("There may be an error as %s file does not exist"%(fnPos))

    def joinSetOfParticlesStep( self, micIds, mode):
        #Create images.xmd metadata
        fnImages = self._getExtraPath("particles_%s.xmd" % mode)
        imgsXmd = MD.MetaData()
        posFiles = glob(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%mode, '*.pos'))

        for posFn in posFiles:
            xmdFn = self._getExtraPath("parts_%s" % mode,
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
        
    def trainCNN(self):

        netDataPath = self._getExtraPath("nnetData")
        makePath(netDataPath)
        nEpochs = self.nEpochs.get()
        if self.doContinue.get():
            prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
            copyTree(prevRunPath, netDataPath)
            if not self.keepTraining.get():
                nEpochs = 0
            
        posTrainDict = {self._getExtraPath("particles_AND.xmd"):  1}
        negTrainDict = {self._getExtraPath("particles_NOISE.xmd"):  1}

        if self.addTrainingData.get() == self.ADD_TRAIN_MODEL:
          negTrainDict[self._getTmpPath("addNegTrainParticles.xmd")]= 1
        
        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        if self.trainPosSetOfParticles.get():
            posTrainFn = self._getExtraPath("trainTrueParticlesSet.xmd")
            posTrainDict[posTrainFn] = self.trainPosWeight.get()
        if self.trainNegSetOfParticles.get():
            negTrainFn = self._getExtraPath("trainFalseParticlesSet.xmd")
            negTrainDict[negTrainFn] = self.trainNegWeight.get()

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
        if not os.path.isdir(netDataPath) and self.doContinue.get():
            prevRunPath = self.continueRun.get()._getExtraPath('nnetData')
            copyTree(prevRunPath, netDataPath)
            
        if self.usesGpu():
            numberOfThreads = None
            gpuToUse = self.getGpuList()[0]
        else:
            numberOfThreads = self.numberOfThreads.get()
            gpuToUse = None

        predictDict = {self._getExtraPath("particles_OR.xmd"): 1}
        if self.doTesting.get() and self.testPosSetOfParticles.get() and self.testNegSetOfParticles.get():
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
        partSet = self._createSetOfParticles()
        readSetOfParticles(self._getPath("particles.xmd"), partSet)
        inputSampling = self.inputCoordinates[0].get().getMicrographs().getSamplingRate()
        partSet.setSamplingRate(self._getDownFactor() * inputSampling)
        boxSize = self._getBoxSize()

        # COORDINATES
        writeSet=False
        if self.checkIfPrevRunIsCompatible("coords_"):
          writeSet=True
        if not "OR" in self.coordinatesDict:
            self.loadCoords(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'OR'),
                                     'OR', writeSet=False)

        coordSet = SetOfCoordinates(filename=self._getPath("coordinates.sqlite"))
        coordSet.copyInfo(self.coordinatesDict['OR'])
        coordSet.setBoxSize(boxSize)
        coordSet.setMicrographs(self.coordinatesDict['OR'].getMicrographs())

        downFactor = self._getDownFactor()
        for part in partSet:
            coord = part.getCoordinate().clone()
            coord.scale(downFactor)

            deepZscoreLabel = '_xmipp_%s' % xmipp.label2Str(MD.MDL_ZSCORE_DEEPLEARNING1)
            setattr(coord, deepZscoreLabel, getattr(part, deepZscoreLabel))
            coordSet.append(coord)
        
        coordSet.write()
        partSet.write()
        
        self._defineOutputs(outputCoordinates=coordSet)        
        self._defineOutputs(outputParticles=partSet)

        for inSetOfCoords in self.inputCoordinates:
            self._defineSourceRelation(inSetOfCoords.get(), coordSet)
            self._defineSourceRelation(inSetOfCoords.get(), partSet)

        print("OR", self.coordinatesDict['OR'].getBoxSize())
#        raise ValueError("peta")

    def _summary(self):
        message = []
        for i, coordinates in enumerate(self.inputCoordinates):
            protocol = self.getMapper().getParent(coordinates.get())
            message.append("Data source  %d %s" % (i + 1, protocol.getClassLabel()))
        message.append("Relative Radius = %f" % self.consensusRadius)
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
