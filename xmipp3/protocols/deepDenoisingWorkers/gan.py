import sys, os
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

from .DeepLearningGeneric import DeepLearningModel
from .dataGenerator import normalization, getDataGenerator


BATCH_SIZE= 32
class GAN(DeepLearningModel):
  def __init__(self,  boxSize, saveModelDir, gpuList="0", batchSize=BATCH_SIZE):
    DeepLearningModel.__init__(self,boxSize, saveModelDir, gpuList, batchSize)

  def _setShape(self, boxSize):
    self.img_rows = boxSize
    self.img_cols = boxSize
    self.channels = 1
    self.shape = self.img_rows * self.img_cols
    self.img_shape = (self.img_rows, self.img_cols, self.channels)
    return self.shape, self.img_shape
    
  def initModelsForTrainining(self):

    # Build the discriminator
    self.discriminator = self.build_discriminator()

    # Build the generator
    self.predictionModelNoParallel = self.build_generator()
    if len(self.gpuList.split(',')) > 1:
      self.predictionModel = multi_gpu_model(self.predictionModelNoParallel)
    else:
      self.predictionModel = self.predictionModelNoParallel

    # The generator takes noise as input and generated imgs
    z = Input(shape=self.img_shape)
    img = self.predictionModel(z)
    # For the combined model we will only train the generator
    self.discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = self.discriminator(img)
    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity
    self.combined = Model(z, valid)
    if len(self.gpuList.split(',')) > 1:
        self.combined = multi_gpu_model(self.combined)
        
  def extractTrainData(self, path, label, norm=-1):

    metadata = xmippLib.MetaData(path)
    Image = []
    I = xmippLib.Image()
    cont = 0
    for itemId in metadata:
      fn = metadata.getValue(label, itemId)
      I.read(fn)
      Imresize = I.getData()

      if norm == -1:
          Imnormalize = 2 * (Imresize - np.min(Imresize)) / (
                  np.max(Imresize) - np.min(
              Imresize)) - 1
      elif norm == 0:
          Imnormalize = Imresize
      elif norm == 1:
          Imnormalize = (Imresize - np.min(Imresize)) / (
                  np.max(Imresize) - np.min(
              Imresize))
      else:
          Imnormalize = (Imresize - np.mean(Imresize)) / np.std(
              Imresize)
      Image.append(Imnormalize)

      if cont > 3000:
          break
      cont += 1

    Image = np.array(Image).astype('float')
    Image = Image.reshape(len(Image), Image.shape[1], Image.shape[2], 1)

    return Image


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

  def generate_data(self, images, batch_size):

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


  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, save_interval=200):
    self.initModelsForTrainining()
    X_train = self.extractTrainData(xmdProjections, xmippLib.MDL_IMAGE, -1)
    batch_size= self.batchSize
    half_batch = int(batch_size / 2)
    self.true, self.noise = self.generate_data(X_train, 100)
    self.true = normalization(self.true, -1, False)
    self.noise = normalization(self.noise, -1, False)
    self.lossD = []
    self.lossG = []
    self.validationMetricList = [99999999999999999]
    lossEpoch = []
    saveImagesPath= os.path.split(xmdParticles)[0]
    #COMPILING MODELS
    optimizer = Adam(learningRate*0.5)
    self.predictionModel.compile(loss='mean_squared_error',
                           optimizer=optimizer)

    self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    nEpochs_init= nEpochs
    nEpochs= save_interval+1+int(nEpochs*10)
    print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
    sys.stdout.flush()                     
    for epoch in range( nEpochs ):

      # ---------------------
      #  Train Discriminator
      # ---------------------
      imgs, noise1 = self.generate_data(X_train, half_batch)
      imgs = normalization(imgs,1, False)
      noise1 = normalization(noise1, -1, False)
      # Select a random half batch of images

      # Generate a half batch of new images
      gen_imgs = self.predictionModel.predict(noise1)
      gen_imgs = normalization(gen_imgs,1, False)
      # Train the discriminator
      d_loss_real = self.discriminator.train_on_batch(imgs,
                                                      np.round(
                                                          np.random.uniform(
                                                              0.9,
                                                              1.0,
                                                              half_batch),
                                                          1))
      d_loss_fake = self.discriminator.train_on_batch(gen_imgs,
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
      imgs2, noise2 = self.generate_data(X_train, batch_size)
      noise2 = normalization(noise2,-1,False)
      # The generator wants the discriminator to label the generated samples
      # as valid (ones)
      valid_y = np.array([1] * batch_size)

      # Train the generator
      g_loss = self.combined.train_on_batch(noise2, valid_y)

      # Plot the progress
      print ("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
          epoch, nEpochs, d_loss[0], 100 * d_loss[1], g_loss))

      evaluate = self.predictionModel.evaluate(self.noise, self.true)
      print "Validation =", evaluate

      if epoch > save_interval and evaluate <= np.min(self.validationMetricList):
          self.predictionModelNoParallel.save(self.saveModelDir)
          self.validationMetricList.append(evaluate)

      self.lossD.append(d_loss[0])
      self.lossG.append(g_loss)
#      self.validationMetricList.append(evaluate)
      lossEpoch.append(d_loss[0])
      # If at save interval => save generated image samples
      if epoch % save_interval == 0:
          print("MeanLoss = ", np.mean(lossEpoch))
          self.save_imgs(saveImagesPath, epoch)
          lossEpoch = []

  def yieldPredictions(self, xmdParticles, xmdProjections=None):
    for batchX, batchY in DeepLearningModel.yieldPredictions(self, xmdParticles, xmdProjections):
      yield normalization(batchX, 1), normalization(batchY, 1)      

  def save_imgs(self, saveImagesPath,  epoch):
    gen_imgs = self.predictionModel.predict(self.noise)
    filename = "denoise_%d.png"
    # Rescale images 0 - 1
    #gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(10, 3)

    axs[0, 0].set_title("reference")
    axs[0, 1].set_title("noisy")
    axs[0, 2].set_title("predicted")
        
    cnt = 0
    for i in range(10):
        axs[i, 0].imshow(self.true[cnt, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(self.noise[cnt, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, 2].axis('off')
        cnt += 1
    plt.savefig( os.path.join(saveImagesPath, filename % epoch))
    plt.close()
    
