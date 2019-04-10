# **************************************************************************
# *
# * Authors:     Javier Mota   (original implementation)
# *              Ruben Sanchez (added U-net and refactoring)
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

import sys, os
import numpy as np

from pyworkflow import VERSION_2_0
from .protocol_generate_reprojections import XmippProtGenerateReprojections
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
import pyworkflow.em.metadata as md
from pyworkflow.utils.path import cleanPath

import xmippLib
from xmipp3.convert import writeSetOfParticles, setXmippAttributes, xmippToLocation
from xmipp3.utils import getMdSize
import xmipp3

from deepDenoisingWorkers.deepDenoising import getModelClass
       
EXEC_MODES= ['Train & Predict','Predict'] 
ITER_TRAIN = 0
ITER_PREDICT = 1

MODEL_TYPES = ["GAN", "U-Net"]
MODEL_TYPE_GAN = 0
MODEL_TYPE_UNET = 1


TRAINING_DATA_MODE = ["ParticlesAndSyntheticNoise", "OnlyParticles"]
TRAINING_DATA_MODE_SYNNOISE = 0
TRAINING_DATA_MODE_PARTS = 1


TRAINING_LOSS = [ "MSE", "PerceptualLoss", "Both"]
TRAINING_LOSS_MSE = 0
TRAINING_LOSS_PERCEPTUAL = 1
TRAINING_LOSS_BOTH = 2

class XmippProtDeepDenoising(XmippProtGenerateReprojections):

    _label ="deep denoising"
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        XmippProtGenerateReprojections.__init__(self, **args)
        
    def _defineParams(self, form):

        form.addSection('Input')
        form.addParam('deepMsg', params.LabelParam, default=True,
                      label='Ensure deepLearningToolkit is installed')
        form.addHidden(params.GPU_LIST, params.StringParam, default='',
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " In case to use several GPUs separate with comas:"
                            "0,1,2")
                            
        form.addParam('modelType', params.EnumParam,
                      choices=MODEL_TYPES,
                      default=MODEL_TYPE_UNET,
                      label='Select model type',
                      help='If you set to *%s*, GAN will be employed '
                           'employed. If you set to *%s* U-Net will be used instead'
                           % tuple(MODEL_TYPES))
                                                       
        form.addParam('modelMode', params.EnumParam, choices=EXEC_MODES,
                       default=ITER_TRAIN,
                       label='Train or predict model',
                       help='*Train*: Train the model using noisy particles '
                            'or their projections in an initial volume'
                            '*Predict*: The particles are denoised with a '
                            'pretrained model')
                            
        form.addParam('inputProjections', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      condition='modelMode==%d'%ITER_TRAIN,

                      label='Input projections of the volume to train',
                      help='use the '
                      'protocol generate reprojections to generate the '
                      'reprojections views')

        form.addParam('modelPretrain', params.BooleanParam, default = False,
                      condition='modelMode==%d'%ITER_PREDICT,
                      label='Choose your '
                      'own model', help='Setting "yes" '
                      'you can choose your own model trained. If you choose'
                      '"no" a general model pretrained will be assign')

        form.addParam('ownModel', params.PointerParam,
                      pointerClass=self.getClassName(),
                      condition='modelPretrain==True and modelMode==%d'%ITER_PREDICT,
                      label='Set your model',
                      help='Choose the protocol where your model is trained')

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      label='Input noisy particles to denoise', help='Input '
                                                                   'noisy '
                      'particles from the protocol generate reprojections if '
                      'you are training or from any other protocol if you are '
                      'predicting')

        form.addParam('imageSize', params.IntParam,
                      label='Scale images to (px)',
                      default=128, help='Scale particles to desired size to improve training'
                                        'The recommended particle size is 128 px. The size must be even.'
                                         'Do not use loss=perceptualLoss or loss=Both if  96< size <150')

        form.addSection(label='Training')
        
        form.addParam('nEpochs', params.FloatParam,
                      label="Number of epochs", default=25.0,
                      help='Number of epochs for neural network training. GAN requires much '
                           'more epochs (>100) to obtain succesfull results')
                      
        form.addParam('learningRate', params.FloatParam,
                      label="Learning rate", default=1e-4,
                      help='Learning rate for neural network training')


        form.addParam('modelDepth', params.IntParam, default=4,
                       condition='modelMode==%d'%ITER_TRAIN, expertLevel=cons.LEVEL_ADVANCED,
                       label='Model depth',
                       help='Indicate the model depth. For 128-64 px images, 4 is the recommend value. '
                            ' larger images may require bigger models') 
                            
        form.addParam('trainingSetType', params.EnumParam, choices=TRAINING_DATA_MODE,
                       default=TRAINING_DATA_MODE_SYNNOISE, expertLevel=cons.LEVEL_ADVANCED,
                       label='Select how to generate training set',
                       help='*ParticlesAndSyntheticNoise*: Train using particles and synthetic noise\n'
                            'or\n*OnlyParticles*: using only particles\n'
                            'or\n*Both*: Train using both strategies')
                          
        form.addParam('trainingLoss', params.EnumParam, choices=TRAINING_LOSS,
                       default=TRAINING_LOSS_BOTH, expertLevel=cons.LEVEL_ADVANCED,
                       label='Select loss for training',
                       help='*MSE*: Train using mean squered error'
                            'or\n*PerceptualLoss*: Train using DeepConsensus perceptual loss\n'
                            'or\n*Both*: Train using both DeepConsensus perceptual loss and mean squered error\n')                                                   
                          
                            
        form.addParam('numberOfDiscVsGenUpdates', params.IntParam, default=5,
                       condition='modelType==%d'%MODEL_TYPE_GAN, expertLevel=cons.LEVEL_ADVANCED,
                       label='D/G trainig ratio',
                       help='Indicate the number of times the discriminator is trained for each '
                            'generator training step. If discriminator loss is going to 0, make it '
                            'smaller, whereas if the discriminator is not training, make it bigger')                           

        form.addParam('loss_logWeight', params.FloatParam, default=3, expertLevel=cons.LEVEL_ADVANCED,
                       condition='modelType==%d'%MODEL_TYPE_GAN,
                       label='D/G loss ratio',
                       help='Indicate the 10^lossRatio times that the generator loss is stronger than '
                            ' the discriminator loss. If discriminator loss is going to 0, make it '
                            'smaller, whereas if the generator is not training, make it bigger')    
                            
        form.addParallelSection(threads=2, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
	
        deps = []
        ret= self._insertFunctionStep('preprocessData', prerequisites=[])
        deps+= [ret]
        if self.modelMode.get() == ITER_TRAIN:
            ret= self._insertFunctionStep('trainModel', prerequisites=deps)
            deps+= [ret]
        ret= self._insertFunctionStep('predictModel', prerequisites=deps)
        deps+= [ret]
        ret= self._insertFunctionStep('createOutputStep', prerequisites=deps)
        deps+= [ret]
        
    def preprocessData(self):
        if self.modelMode.get() == ITER_PREDICT and self.modelType.get() == MODEL_TYPE_UNET and not self.modelPretrain:
          raise ValueError("Predict directly with UNET is not implemented yet")
          
        self.particles = self._getExtraPath('noisyParticles.xmd')
        writeSetOfParticles(self.inputParticles.get(), self.particles)
        self.metadata = xmippLib.MetaData(self.particles)
        fnNewParticles = self._getExtraPath('resizedParticles.stk')
        self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
            self.particles, fnNewParticles, self.imageSize.get()))

        if self.modelMode.get() == ITER_TRAIN:
            projections = self._getExtraPath('projections.xmd')
            writeSetOfParticles(self.inputProjections.get(), projections)
            fnNewProjections = self._getExtraPath('resizedProjections.stk')
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
                projections, fnNewProjections, self.imageSize.get()))


    def trainModel(self):

        modelFname = self._getPath('ModelTrained.h5')
        ModelClass= getModelClass( MODEL_TYPES[self.modelType.get()], self.gpuList.get())
         
        builder_args= {"boxSize":self.imageSize.get(), "saveModelFname":modelFname, 
                       "modelDepth": self.modelDepth.get(), "gpuList":self.gpuList.get(),
                       "generatorLoss": TRAINING_LOSS[self.trainingLoss.get()],
                       "trainingDataMode": TRAINING_DATA_MODE[self.trainingSetType.get()] }
        if self.modelType.get() == MODEL_TYPE_GAN:

          builder_args["training_DG_ratio"]= self.numberOfDiscVsGenUpdates.get()
          builder_args["loss_logWeight"]= self.loss_logWeight.get()
          
        model = ModelClass( **builder_args )
        
        dataPathParticles= self._getExtraPath('resizedParticles.xmd')
        dataPathProjections= self._getExtraPath('resizedProjections.xmd')
                    
        model.train( self.learningRate.get(), self.nEpochs.get(), dataPathParticles, dataPathProjections )
        model.clean()
        del model
#        raise ValueError("training ended")
        
    def predictModel(self):
        from scipy.stats import pearsonr
        
        if self.modelMode.get() == ITER_PREDICT:
            if self.modelPretrain == True:
                modelFname = self.ownModel.get()._getPath('ModelTrained.h5')
            else:
                modelFname = xmipp3.Plugin.getModel('deepDenoising', 'PretrainModel.h5')
        else:
            modelFname = self._getPath('ModelTrained.h5')
                
        ModelClass= getModelClass( MODEL_TYPES[self.modelType.get()], self.gpuList.get())

        builder_args= {"boxSize":self.imageSize.get(), "saveModelFname":modelFname, 
                       "modelDepth": self.modelDepth.get(), "gpuList":self.gpuList.get(),
                       "generatorLoss": TRAINING_LOSS[self.trainingLoss.get()], "batchSize": 2000}
                       
        model = ModelClass( **builder_args)
        
        inputParticlesMdName= self._getExtraPath('resizedParticles.xmd' )
        inputParticlesStackName= self._getExtraPath('resizedParticles.stk' )
        outputParticlesStackName= self._getExtraPath('particlesDenoised.stk' )
        outputParticlesMdName= self._getExtraPath('particlesDenoised.xmd' )
        inputProjectionsStackName= self._getExtraPath('resizedProjections.stk' )
           
        #metadataParticles = xmippLib.MetaData(inputParticlesMdName )
        if self.modelMode.get() == ITER_TRAIN:
            metadataProjections = xmippLib.MetaData(inputProjectionsStackName )

        dimMetadata = getMdSize(inputParticlesMdName )
        xmippLib.createEmptyFile(outputParticlesStackName, self.imageSize.get(),
                                                   self.imageSize.get(),1, dimMetadata)

        mdNewParticles = md.MetaData()
        
        I = xmippLib.Image()
        i=1 #TODO. Is this the correct way? Should we use particle ids instead
        
        for preds, particles, projections in model.yieldPredictions(inputParticlesMdName, metadataProjections 
                                                                          if self.modelMode.get() == ITER_TRAIN else None ):
          newRow = md.Row()
          for pred, particle, projection in zip(preds, particles, projections):
              outputImgpath = ('%06d@' %(i ,)) + outputParticlesStackName
              I.setData(np.squeeze(pred))
              I.write(outputImgpath)

              pathNoise = ('%06d@' %(i,)) + inputParticlesStackName

              newRow.setValue(md.MDL_IMAGE, outputImgpath)
              newRow.setValue(md.MDL_IMAGE_ORIGINAL, pathNoise)
              if self.modelMode.get() == ITER_TRAIN:
                  pathProj = ('%06d@' %(i ,)) + inputProjectionsStackName
                  newRow.setValue(md.MDL_IMAGE_REF, pathProj)
                  correlations1, _ = pearsonr(pred.ravel(), projection.ravel())
                  newRow.setValue(md.MDL_CORR_DENOISED_PROJECTION, correlations1)
              newRow.addToMd(mdNewParticles)
              i+=1

        mdNewParticles.write('particles@' + outputParticlesMdName, xmippLib.MD_APPEND)
        self.runJob("xmipp_transform_normalize", "-i %s --method NewXmipp "
                    "--background circle %d "%(outputParticlesStackName, self.imageSize.get()/2))
                    
    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        Ts = imgSet.getSamplingRate()
        xdim = imgSet.getDimensions()[0]
        outputSet.setSamplingRate((Ts*xdim)/self.imageSize.get())
        imgFn = self._getExtraPath('particlesDenoised.xmd')
        outputSet.copyItems(imgSet,
                            updateItemCallback=self._processRow,
                            itemDataIterator=md.iterRows(imgFn,
                                                         sortByLabel=md.MDL_ITEM_ID)
                            )
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        cleanPath(self._getExtraPath('resizedParticles.xmd'))
        cleanPath(self._getExtraPath('noisyParticles.xmd'))
        cleanPath(self._getExtraPath('projections.xmd'))
        if os.path.exists(self._getExtraPath('resizedProjections.xmd')):
            cleanPath(self._getExtraPath('resizedProjections.xmd'))

    def _processRow(self, particle, row):
        particle.setLocation(xmippToLocation(row.getValue(xmippLib.MDL_IMAGE)))
        if self.modelMode.get() == ITER_TRAIN:
            setXmippAttributes(particle, row,
                               xmippLib.MDL_CORR_DENOISED_PROJECTION)


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Particles denoised")
        return summary


