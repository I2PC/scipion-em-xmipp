#################################################################################################################
#
#    MODIFIED FROM https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5
#    by rsanchez@cnb.csic.es
#
#################################################################################################################

import sys, os
import random, math
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Activation, \
                         Lambda, Dropout, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K

from skimage.transform import rotate

import xmippLib
import matplotlib.pyplot as plt
import pyworkflow.em.metadata as md
from .DeepLearningGeneric import DeepLearningModel
from .dataGenerator import normalization, getDataGenerator, extractNBatches

from .unet import build_UNet

BATCH_SIZE= 32
CHECKPOINT_AT= 2
NUM_BATCHES_PER_EPOCH= 25

class GAN(DeepLearningModel):
  def __init__(self,  boxSize, saveModelDir, gpuList="0", batchSize=BATCH_SIZE, trainingDataMode="Synthetic"):
    DeepLearningModel.__init__(self, boxSize, saveModelDir, gpuList, batchSize)
    self.epochSize= NUM_BATCHES_PER_EPOCH
    self.trainingDataMode= trainingDataMode
    
  def _setShape(self, boxSize):
    self.img_rows = boxSize
    self.img_cols = boxSize
    self.channels = 1
    self.shape = self.img_rows * self.img_cols
    self.img_shape = (self.img_rows, self.img_cols, self.channels)
    return self.shape, self.img_shape

    

  def buildGenerator(self):

    return build_UNet( self.img_shape, out_ch=1, start_ch=32, depth=3, inc_rate=2., activation='relu', 
                        dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False)
                        
  def buildDiscriminator(self):
    return build_discriminator( self.img_shape )

  def buildCombinedModel(self, generator, discriminator):
    inputs = Input(shape=self.img_shape)
    generated_imgs = generator(inputs)
    isFakePred = discriminator(generated_imgs)
    model = Model(inputs=inputs, outputs=[generated_imgs, isFakePred])
    return model
  
  def getGeneratorLoss(self):
    return keras.losses.mean_squared_error # generatePerceptualLoss(self.img_shape) #keras.losses.mean_squared_error
    
  def getDiscrimLoss(self):
    def wasserstein_loss(y_true, y_pred):
      return K.mean(y_true*y_pred)
            
    return wasserstein_loss
    
  def getRandomRows(self, x, n):
      return x[np.random.choice(x.shape[0], n),... ]

  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, save_interval=CHECKPOINT_AT):
  
    discriminator = self.buildDiscriminator()

    # Build the generator
    generatorModel = self.buildGenerator()
    combinedModel = self.buildCombinedModel(generatorModel, discriminator)
    N_GPUs= len(self.gpuList.split(','))
    if N_GPUs > 1:
        combinedModel = multi_gpu_model(combinedModel, gpus= N_GPUs)
        
    optimizer_discriminator=  Adam(lr=learningRate*0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer_combined= Adam(lr=learningRate*0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    # Compile models
    discriminator.trainable = True
    discriminator.compile(optimizer=optimizer_discriminator, loss=self.getDiscrimLoss())
    discriminator.trainable = False
    loss = [self.getGeneratorLoss(), self.getDiscrimLoss()]
    loss_weights = [100, 1]
    
    combinedModel.compile(optimizer=optimizer_combined, loss=loss, loss_weights=loss_weights)
    discriminator.trainable = True
        
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=True, valFraction=0.1,
                                 augmentData=False, batchSize=min(self.batchSize, md.MetaData(xmdParticles).size()))   

    nEpochs_init= nEpochs
    nEpochs= int(max(save_interval+1, nEpochs_init*float(stepsPerEpoch)/self.epochSize ))
    print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
    sys.stdout.flush()
    

    valIterator, valStepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=False, valFraction=0.1,
                                           augmentData=False, nEpochs= 1, batchSize=100 )

    particles_val, projections_val = extractNBatches(valIterator, 2)
    del valIterator

    bestValidationLoss = sys.maxsize
    saveImagesPath= os.path.split(xmdParticles)[0]
    saveImagesPath= os.path.join(saveImagesPath, "batchImages")
    if not os.path.exists(saveImagesPath):
      os.mkdir(saveImagesPath)

    roundsToEarlyStopping=10
    roundsToLR_decay=3

    roundsNoImprovement=0
    for epoch in range(nEpochs):
      discriminatorLoss_list=[]
      generatorLoss_list=[]
      for batchNum, (X_particles, Y_projections) in enumerate( trainIterator ):
    
        generated_imgs= generatorModel.predict(X_particles, batch_size= self.batchSize, verbose=0)
        
        X_forDiscriminator= np.concatenate([Y_projections, generated_imgs])
        Y_forDiscriminator= np.zeros( X_forDiscriminator.shape[:1] )
        Y_forDiscriminator[:Y_projections.shape[0] ]=1

        discriminatorLoss= discriminator.train_on_batch(X_forDiscriminator, Y_forDiscriminator)
        discriminatorLoss_list.append(discriminatorLoss)
        
        discriminator.trainable = False 
        __, generatorLoss, __ = combinedModel.train_on_batch(X_particles, [Y_projections, np.ones( Y_projections.shape[:1]) ])
        generatorLoss_list.append(generatorLoss)
        discriminator.trainable = True
        

        if batchNum>= self.epochSize:
          break
              
      combinedValLoss, generatorValLoss, __ = combinedModel.evaluate(particles_val, 
                                                    [projections_val, np.ones( projections_val.shape[:1]) ], verbose=0)
                                                    
      # Plot the progress at the end of each epoch      
      print("\nEpoch %d/%d ended. discr_loss= %2.5f  generator_loss= %2.5f  val_generator_loss= %2.5f"%(epoch+1,nEpochs,
                                           np.mean(discriminatorLoss_list), np.mean(generatorLoss_list), generatorValLoss) )

      if generatorValLoss <= bestValidationLoss:
          print("Saving model. validation meanLoss improved from %2.6f to %2.6f"%(bestValidationLoss, generatorValLoss ) )
          generatorModel.save(self.saveModelDir)
          bestValidationLoss= generatorValLoss
          roundsNoImprovement= 0
      else:
          print("Validation meanLoss did not improve from %s"%(bestValidationLoss ) )
          roundsToEarlyStopping-=1
          roundsNoImprovement+=1
      sys.stdout.flush()
      
      if roundsNoImprovement== roundsToLR_decay:
        new_lr= 01.* learningRate
        print("Decreasing learning rate to %f"%(learningRate) )
        K.set_value(optimizer_discriminator.lr, new_lr)
        K.set_value(optimizer_combined.lr, new_lr)
        learningRate= new_lr
        
      # If at save interval => save generated image samples
      if epoch % save_interval == 0:
          gen_imgs = generatorModel.predict(particles_val, verbose=0)
          self.save_imgs(gen_imgs, particles_val, projections_val, saveImagesPath, epoch)
          
      if epoch>= nEpochs:
        break
      elif roundsToEarlyStopping<0:
        print("Early stopping")
      print("----------------------------------------------------")
      
  def yieldPredictions(self, xmdParticles, xmdProjections=None):
    print("applying denoising"); sys.stdout.flush()
    for batchX, batchY in DeepLearningModel.yieldPredictions(self, xmdParticles, xmdProjections):
      yield normalization(batchX, 1), normalization(batchY, 1)      

  def save_imgs(self, gen_imgs, X, Y, saveImagesPath,  epoch):

    fig, axs = plt.subplots(10, 3)
    
    axs[0, 0].set_title("predicted")
    axs[0, 1].set_title("particles")
    axs[0, 2].set_title("projections")
        
    cnt = 0
    for i in range(10):
        axs[i, 0].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(X[cnt, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(Y[cnt, :, :, 0], cmap='gray')
        axs[i, 2].axis('off')
        cnt += 1

    fname= os.path.join(saveImagesPath, "denoise_%d.png"% epoch)
    if os.path.exists(fname):
      try:
        os.remove(fname)
      except IOError, OSError:
        pass
    plt.savefig( fname)
    plt.close()
    


def build_discriminator( img_shape, nConvLayers= 4):

  model = Sequential()
  assert math.log(img_shape[1],2)> nConvLayers, "Error, too small images: input %s. Min size"%(img_shape, 2**nConvLayers) 
  model.add( Conv2D(2**4,3, activation="linear", padding="same", input_shape= img_shape) )
  for i in range(nConvLayers-1):
    model.add( Conv2D(2**(5+i),3, activation="linear", padding="same") )
    model.add( BatchNormalization() )
    model.add( LeakyReLU(0.1) )
    model.add( MaxPooling2D() )
  model.add(Flatten())
  model.add(Dense( min(256, np.prod(img_shape)) ) )
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dense( min(128, np.prod(img_shape)//2) ) )
  model.add(LeakyReLU(alpha=0.2))

  model.add(Dense(1, activation='sigmoid'))

  img = Input(shape=img_shape)
  validity = model(img)

  return Model(img, validity)
    
    
    

def generatePerceptualLoss(image_shape):
  from keras.applications.vgg16 import VGG16
  import tensorflow as tf
  
  def perceptual_loss(y_true, y_pred):
    input_tensor= Input(image_shape[:-1]+(1,))
    color_tensor = Lambda(lambda image: tf.image.grayscale_to_rgb(image) )(input_tensor)
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor= color_tensor)
    targetLayer='block3_conv3'
    nToPop=0
    for layer in reversed(vgg.layers):
      if layer.name == targetLayer: break
      nToPop+=1
    for i in range(nToPop):
      vgg.layers.pop()

    loss_model = Model(inputs=input_tensor, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
  return perceptual_loss
  
