import sys, os
import random
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Activation, \
                         Conv2DTranspose, Dropout, Flatten, Dense

from keras.optimizers import Adam               
from skimage.transform import rotate

import xmippLib
import matplotlib.pyplot as plt
import pyworkflow.em.metadata as md
from .DeepLearningGeneric import DeepLearningModel
from .dataGenerator import normalization, getDataGenerator, extractNBatches


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

  def addNoise(self, image):

    levelsNoise = np.arange(0.1, 1.0, 0.1)
    k = np.random.randint(0, len(levelsNoise))
    noise = np.random.normal(0.0, levelsNoise[k], image.shape)
    imageNoise = image + noise

    return imageNoise

  def applyTransform(self, image):

    angle = np.random.randint(-180, 180)
    imRotate = rotate(image, angle, mode='wrap')
    return imRotate

  def generate_synt_data_fromProj(self, images, batch_size):

    proj = []
    noiseImage = []
    for j in range(0, batch_size):
      idx = np.random.randint(0, images.shape[0])
      img = images[idx]

      projection = self.applyTransform(img)
      noise = self.addNoise(projection)
      proj.append(projection)
      noiseImage.append(noise)

    projections = np.asarray(proj).astype('float32')
    imageNoise = np.asarray(noiseImage).astype('float32')

    return projections, imageNoise
    

  def build_generator(self):

    input_img = Input(shape=self.img_shape,
                      name='input')
    x = Conv2D(64, (5, 5), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(16, kernel_size=5, strides=1, padding='same')(
        x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=1, padding='same')(
        x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(
        x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=1, strides=1, padding='same')(
        x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(1,kernel_size=1,strides=1, padding='same')(x)
    decoded = Activation('linear')(x)

    model = Model(input_img, decoded)

    noise = Input(shape=self.img_shape)
    img = model(noise)

    return Model(noise, img)

  def build_discriminator(self):

    img_shape = (self.img_rows, self.img_cols, self.channels)

    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(self.img_rows*self.img_cols))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense((self.img_rows*self.img_cols)/2))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

  def getRandomRows(self, x, n):
      return x[np.random.choice(x.shape[0], n),... ]

  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, save_interval=CHECKPOINT_AT):
  
    discriminator = self.build_discriminator()

    # Build the generator
    predictionModelNoParallel = self.build_generator()
    if len(self.gpuList.split(',')) > 1:
      predictionModel = multi_gpu_model(predictionModelNoParallel)
    else:
      predictionModel = predictionModelNoParallel

    # The generator takes noise as input and generated imgs
    z = Input(shape=self.img_shape)
    img = predictionModel(z)
    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(img)
    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity
    combined = Model(z, valid)
    if len(self.gpuList.split(',')) > 1:
        combined = multi_gpu_model(combined)
        
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=True, valFraction=0.1,
                                 augmentData=False, batchSize=min(3000, md.MetaData(xmdParticles).size()))   

    nEpochs_init= nEpochs
    nEpochs= int(max(save_interval+1, nEpochs_init*float(stepsPerEpoch)/self.epochSize ))
    print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
    sys.stdout.flush()
    
#    X_train = self.extractTrainData(xmdProjections, xmippLib.MDL_IMAGE, -1)

    valIterator, valStepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=False, valFraction=0.1,
                                           augmentData=False, nEpochs= 1, batchSize=100 )
    batch_size= self.batchSize
    half_batch = int(batch_size / 2)

    particles_val, projections_val = extractNBatches(valIterator, 2)
    del valIterator
    trueData, noisedData = self.generate_synt_data_fromProj(projections_val, 2*100)
#    trueData, noisedData = self.generate_synt_data_fromProj(X_train, 100)
    trueData = normalization(trueData, -1, False)
    noisedData = normalization(noisedData, -1, False)
    lossD = []
    lossG = []
    bestValidationLoss = 99999999999999999
    lossEpoch = []
    saveImagesPath= os.path.split(xmdParticles)[0]
    #COMPILING MODELS
    optimizer = Adam(learningRate*0.5)
    predictionModel.compile(loss='mean_squared_error',
                           optimizer=optimizer)

    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])       
    
    roundsToEearlyStopping=10

    if self.trainingDataMode== "Both":
      selectProbas=[0.5,0.5]
    elif self.trainingDataMode== "SyntheticFromProjections":
      selectProbas=[1,0]
    elif self.trainingDataMode== "Particles":
      selectProbas=[0,1]
      
    for epoch, (X_particles, X_projections) in enumerate( trainIterator ):
      for batch in range(NUM_BATCHES_PER_EPOCH):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        randNum= np.random.rand()
        if randNum< 1e-10+selectProbas[0]:
          imgs, noise1 = self.generate_synt_data_fromProj(X_projections, half_batch)
        else:
          imgs, noise1= self.getRandomRows(X_projections, half_batch), self.getRandomRows(X_particles, half_batch)
          
        imgs = normalization(imgs,1, False)
        noise1 = normalization(noise1, -1, False)
        # Select a random half batch of images

        # Generate a half batch of new images
        gen_imgs = predictionModel.predict(noise1, verbose=0)
        gen_imgs = normalization(gen_imgs,1, False)
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs,
                                                        np.round(
                                                            np.random.uniform(
                                                                0.9,
                                                                1.0,
                                                                half_batch),
                                                            1))
        d_loss_fake = discriminator.train_on_batch(gen_imgs,
                                                        np.round(
                                                            np.random.uniform(
                                                                0.0,
                                                                0.1,
                                                                half_batch),
                                                            1))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Generator
        # ---------------------
        
        if randNum< 1e-10+selectProbas[0]:
          imgs2, noise2 = self.generate_synt_data_fromProj(X_projections, batch_size)
        else:
          imgs2, noise2= self.getRandomRows(X_projections, batch_size), self.getRandomRows(X_particles, batch_size)
        
        noise2 = normalization(noise2,-1,False)
        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * batch_size)

        # Train the generator
        g_loss = combined.train_on_batch(noise2, valid_y)

        # Plot the progress
      print ("epoch: %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
          epoch, nEpochs, d_loss[0], 100 * d_loss[1], g_loss))
      evaluate = predictionModel.evaluate(noisedData, trueData, verbose=0)
      print("Epoch end. Validation metrics = %s"%evaluate)

      if evaluate <= bestValidationLoss:
          print("Saving model. validation meanLoss improved from %s to %s"%(bestValidationLoss, evaluate ) )
          predictionModelNoParallel.save(self.saveModelDir)
          bestValidationLoss= evaluate
      else:
          print("Validation meanLoss did not improve from %s"%(bestValidationLoss ) )
          roundsToEearlyStopping-=1
      sys.stdout.flush()
      
      lossD.append(d_loss[0])
      lossG.append(g_loss)
      lossEpoch.append(d_loss[0])
      # If at save interval => save generated image samples
      if epoch % save_interval == 0:
          self.save_imgs(predictionModel, noisedData, trueData, saveImagesPath, epoch)
          lossEpoch = []
          
      if epoch>= nEpochs:
        break
      elif roundsToEearlyStopping<0:
        print("Early stopping")
      print("----------------------------------------------------")
      
  def yieldPredictions(self, xmdParticles, xmdProjections=None):
    print("applying denoising"); sys.stdout.flush()
    for batchX, batchY in DeepLearningModel.yieldPredictions(self, xmdParticles, xmdProjections):
      yield normalization(batchX, 1), normalization(batchY, 1)      

  def save_imgs(self, predictionModel, X, Y, saveImagesPath,  epoch):
    gen_imgs = predictionModel.predict(X)
    filename = "denoise_%d.png"
    # Rescale images 0 - 1
    #gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(10, 3)

    axs[0, 0].set_title("reference")
    axs[0, 1].set_title("noisy")
    axs[0, 2].set_title("predicted")
        
    cnt = 0
    for i in range(10):
        axs[i, 0].imshow(Y[cnt, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(X[cnt, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, 2].axis('off')
        cnt += 1
    plt.savefig( os.path.join(saveImagesPath, filename % epoch))
    plt.close()
    
