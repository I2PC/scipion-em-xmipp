import sys
import keras
import math

from dataGenerator import getDataGenerator,  BATCH_SIZE


class DeepLearningModel():
  def __init__(self, boxSize, saveModelDir, gpuList, batchSize):
  
    self.saveModelDir = saveModelDir
    self.gpuList= gpuList
    self.batchSize= batchSize
    self._setShape(boxSize)
    
  def _setShape(self, boxSize):
    raise ValueError("Not implemented yet")
    
  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections):        
    raise ValueError("Not implemented yet")
                        

  def yieldPredictions(self, xmdParticles, xmdProjections=None):
         
    if xmdProjections is None:
      xmdProjections= xmdParticles
    model = keras.models.load_model(self.saveModelDir, custom_objects={})
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdParticles, isTrain=False, 
                                           valFraction=0, augmentData=False, nEpochs=1, batchSize= self.batchSize)
    for batchX, batchY in trainIterator:
      yield model.predict(batchX, batch_size=BATCH_SIZE), batchY

