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
import matplotlib.pyplot as plt

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
        
ITER_TRAIN = 0
ITER_PREDICT = 1

MODEL_TYPES = ["GAN", "U-Net"]
MODEL_TYPE_GAN = 0
MODEL_TYPE_UNET = 1

class XmippProtDeepDenoising(XmippProtGenerateReprojections):

    _label ="deep denoising"
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        XmippProtGenerateReprojections.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL
        
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
                      default=MODEL_TYPE_GAN,
                      label='Select model type',
                      help='If you set to *%s*, GAN will be employed '
                           'employed. If you set to *%s* U-Net will be used instead'
                           % tuple(MODEL_TYPES))
                                                       
        form.addParam('modelMode', params.EnumParam, choices=['Train & Predict',
                                                          'Predict'],
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
                      label='Images size',
                      default=64, help='It is recommended to use small sizes '
                                        'to have a faster training. The size '
                                        'must be even. Using high sizes '
                                        'several GPUs are required' )


        form.addSection(label='Training')
        
        form.addParam('nEpochs', params.FloatParam,
                      label="Number of epochs", default=5.0,
                      help='Number of epochs for neural network training.')
                      
        form.addParam('learningRate', params.FloatParam,
                      label="Learning rate", default=1e-4,
                      help='Learning rate for neural network training')


        form.addParallelSection(threads=1, mpi=5)

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
        
        model = ModelClass( self.imageSize.get(), modelFname, self.gpuList.get())
        
        dataPathParticles= self._getExtraPath('resizedParticles.xmd')
        dataPathProjections= self._getExtraPath('resizedProjections.xmd')

        model.train( self.learningRate.get(), self.nEpochs.get(), dataPathParticles, dataPathProjections )


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
        
        model = ModelClass( self.imageSize.get(), modelFname, self.gpuList.get(), batchSize= 2000)
        
        inputParticlesMdName= self._getExtraPath('resizedParticles.xmd' )
        inputParticlesStackName= self._getExtraPath('resizedParticles.stk' )
        outputParticlesStackName= self._getExtraPath('particlesDenoised.stk' )
        outputParticlesMdName= self._getExtraPath('particlesDenoised.xmd' )
        inputProjectionsStackName= self._getExtraPath('resizedProjections.stk' )
           
        metadataParticles = xmippLib.MetaData(inputParticlesMdName )
        metadataProjections = xmippLib.MetaData(inputProjectionsStackName )

        dimMetadata = getMdSize(inputParticlesMdName )
        xmippLib.createEmptyFile(outputParticlesStackName, self.imageSize.get(),
                                                   self.imageSize.get(),1, dimMetadata)

        mdNewParticles = md.MetaData()
        
        I = xmippLib.Image()
        i=1 #TODO. Is this the correct way? Should we use particle ids instead
        
        for imgPred_batch, imgProjection_batch in model.yieldPredictions(inputParticlesMdName, metadataProjections 
                                                                          if self.modelMode.get() == ITER_TRAIN else None ):
          newRow = md.Row()
          for img, projection in zip(imgPred_batch, imgProjection_batch):
              outputImgpath = ('%06d@' %(i ,)) + outputParticlesStackName
              I.setData(np.squeeze(img))
              I.write(outputImgpath)

              pathNoise = ('%06d@' %(i,)) + inputParticlesStackName

              newRow.setValue(md.MDL_IMAGE, outputImgpath)
              newRow.setValue(md.MDL_IMAGE_ORIGINAL, pathNoise)
              if self.modelMode.get() == ITER_TRAIN:
                  pathProj = ('%06d@' %(i ,)) + inputProjectionsStackName
                  newRow.setValue(md.MDL_IMAGE_REF, pathProj)
                  correlations1, _ = pearsonr(img.ravel(), projection.ravel())
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


