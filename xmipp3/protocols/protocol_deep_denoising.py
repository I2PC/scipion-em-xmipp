# **************************************************************************
# *
# * Authors:     Javier Mota   (original implementation)
# *              Ruben Sanchez (added U-net)
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


def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)
        
BAD_IMPORT_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearnigToolkit
If gpu version of tensorflow desired, install cuda 8.0 or cuda 9.0
We will try to automatically install cudnn, if unsucesfully, install cudnn and add to LD_LIBRARY_PATH
add to SCIPION_DIR/config/scipion.conf
CUDA = True
CUDA_VERSION = 8.0  or 9.0
CUDA_HOME = /path/to/cuda-%(CUDA_VERSION)
CUDA_BIN = %(CUDA_HOME)s/bin
CUDA_LIB = %(CUDA_HOME)s/lib64
CUDNN_VERSION = 6 or 7
'''
        
ITER_TRAIN = 0
ITER_PREDICT = 1

ADD_MODEL_TYPES = ["GAN", "U-Net"]
ADD_MODEL_GAN = 0
ADD_MODEL_UNET = 1

class XmippProtDeepDenoising(XmippProtGenerateReprojections):

    _label ="deep denoising"
    _lastUpdateVersion = VERSION_2_0

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
                      choices=ADD_MODEL_TYPES,
                      default=ADD_MODEL_GAN,
                      label='Select model type',
                      help='If you set to *%s*, GAN will be employed '
                           'employed. If you set to *%s* U-Net will be used instead'
                           % tuple(ADD_MODEL_TYPES))
                                                       
        form.addParam('modelMode', params.EnumParam, choices=['Train & Predict',
                                                          'Predict'],
                       default=ITER_TRAIN,
#                       condition='modelType==%d'%ADD_MODEL_GAN,
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
	

        self.newXdim = self.imageSize.get()
        self._insertFunctionStep('preprocessData')
        if self.modelMode.get() == ITER_TRAIN:
            self._insertFunctionStep('trainModel')
        self._insertFunctionStep('predictModel')
        self._insertFunctionStep('createOutputStep')

    def preprocessData(self):
        if self.modelMode.get() == ITER_PREDICT and self.modelType == ADD_MODEL_UNET:
          raise ValueError("Predict directly with UNET not implemented")
        self.particles = self._getExtraPath('noisyParticles.xmd')
        writeSetOfParticles(self.inputParticles.get(), self.particles)
        self.metadata = xmippLib.MetaData(self.particles)
        fnNewParticles = self._getExtraPath('resizedParticles.stk')
        self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
            self.particles, fnNewParticles, self.newXdim))

        if self.modelMode.get() == ITER_TRAIN:
            projections = self._getExtraPath('projections.xmd')
            writeSetOfParticles(self.inputProjections.get(), projections)
            fnNewProjections = self._getExtraPath('resizedProjections.stk')
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
                projections, fnNewProjections, self.newXdim))


    def trainModel(self):
        updateEnviron(self.gpuList.get())
        try:
          import keras
        except ImportError as e:
          print(e)
          raise ValueError(BAD_IMPORT_MSG)
        from keras.models import load_model

        if self.modelType== ADD_MODEL_GAN:
          self.model = GAN()
        else:
          self.model = UNET()
          
        modelFname = self._getPath('ModelTrained.h5')

        self.model.initModel(self.learningRate.get(), self.imageSize.get(), modelFname, self.gpuList.get() )
        
        dataPathProjections= self._getExtraPath('resizedProjections.xmd')
        if self.modelType== ADD_MODEL_GAN:
          self.X_train = self.model.extractTrainData(dataPathProjections, xmippLib.MDL_IMAGE, -1)
          self.model.train(self.X_train, epochs=int(self.nEpochs.get()), batch_size=32, save_interval=200)
        else:
          self.model.train( self.nEpochs.get(), self._getExtraPath('resizedParticles.xmd'), 
                            self._getExtraPath('resizedProjections.xmd') )

        
    def predictModel(self):
        updateEnviron(self.gpuList.get())
        try:
          import keras
        except ImportError as e:
          print(e)
          raise ValueError(BAD_IMPORT_MSG)
        from keras.models import load_model
        self.predictionModel = load_model(self._getPath('ModelTrained.h5'))
        metadataPart = xmippLib.MetaData(self._getExtraPath('resizedParticles.xmd'))
        if self.modelMode.get() == ITER_TRAIN:
            metadataProj = xmippLib.MetaData(self._getExtraPath('resizedProjections.xmd'))
        img = xmippLib.Image()
        dimMetadata = getMdSize(self._getExtraPath('resizedParticles.xmd'))
        xmippLib.createEmptyFile(self._getExtraPath('particlesDenoised.stk'),self.newXdim, self.newXdim,1, dimMetadata)

        mdNewParticles = md.MetaData()

        self.groupParticles = 5000

        if self.modelMode.get() == ITER_PREDICT:
            if self.modelModePretrain:
                model = self.ownModel.get()._getPath('ModelTrained.h5')
                model = load_model(model)
            else:
                myModelfile = xmipp3.Plugin.getModel('deepDenoising', 'PretrainModel.h5')
                model = load_model(myModelfile)
        for num in range(1,dimMetadata,self.groupParticles):
            self.noisyParticles = self.model.extractInfoMetadata(metadataPart,
                                    xmippLib.MDL_IMAGE, img, num, self.groupParticles, -1)

            if self.modelMode.get() == ITER_TRAIN:
                self.predict = self.model.predict(self.predictionModel, self.noisyParticles)
                self.projections = self.model.extractInfoMetadata(metadataProj,
                                    xmippLib.MDL_IMAGE, img, num, self.groupParticles, 1)
            else:
                self.predict = self.model.predict(model, self.noisyParticles)

            self.noisyParticles = self.model.normalization(self.noisyParticles, 1)
            self.prepareMetadata(mdNewParticles, img, num)

        mdNewParticles.write('particles@' + self._getExtraPath('particlesDenoised.xmd'), xmippLib.MD_APPEND)
        self.runJob("xmipp_transform_normalize", "-i %s --method NewXmipp "
                    "--background circle %d "%(self._getExtraPath('particlesDenoised.stk'), self.newXdim/2))

    def prepareMetadata(self, metadata, image, initItem):
        newRow = md.Row()
        for i, img in enumerate(self.predict):
            image.setData(np.squeeze(img))
            path = '%06d@' % (i + initItem) + self._getExtraPath(
                   'particlesDenoised.stk')
            image.write(path)

            pathNoise = '%06d@' % (i + initItem) + self._getExtraPath(
                'resizedParticles.stk')

            newRow.setValue(md.MDL_IMAGE, path)
            newRow.setValue(md.MDL_IMAGE_ORIGINAL, pathNoise)
            if self.modelMode.get() == ITER_TRAIN:
                from scipy.stats import pearsonr
                pathProj = '%06d@' % (i + initItem) + self._getExtraPath(
                    'resizedProjections.stk')
                newRow.setValue(md.MDL_IMAGE_REF, pathProj)
                correlations1, _ = pearsonr(img.ravel(), self.projections[
                    i].ravel())
                newRow.setValue(md.MDL_CORR_DENOISED_PROJECTION, correlations1)
            newRow.addToMd(metadata)


    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        Ts = imgSet.getSamplingRate()
        xdim = imgSet.getDimensions()[0]
        outputSet.setSamplingRate((Ts*xdim)/self.newXdim)
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


class GAN(XmippProtDeepDenoising):

    def initModel(self, learningRate, boxSize, saveModelDir, numGPUs):
        try:
          import keras
        except ImportError as e:
          print(e)
          raise ValueError(BAD_IMPORT_MSG)
        from keras.models import Model
        from keras.layers import Input
        from keras.optimizers import Adam
        from keras.utils import multi_gpu_model

        self.dir2 = saveModelDir
#        optimizer = Adam(0.00005)
        optimizer = Adam(learningRate)
        self.img_rows = boxSize
        self.img_cols = boxSize
        self.channels = 1
        self.shape = self.img_rows * self.img_cols
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # Build and compile the generator
        self.predictionModelNoParallel = self.build_generator()
        if len(numGPUs.split(',')) > 1:
            self.predictionModel = multi_gpu_model(self.predictionModelNoParallel)
        else:
            self.predictionModel = self.predictionModelNoParallel
        self.predictionModel.compile(loss='mean_squared_error',
                               optimizer=optimizer)
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
        if len(numGPUs.split(',')) > 1:
            self.combined = multi_gpu_model(self.combined)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)



    def extractInfoMetadata(self, metadata, label, I, numOfParticles,
                            group, norm=-1):
        Image = []
        numOfParticles = np.arange(numOfParticles, numOfParticles+group)
        try:
            for itemId in numOfParticles:
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
        except:
            print "ALL IMAGES HAVE BEEN READ"

        Image = np.array(Image).astype('float')
        Image = Image.reshape(len(Image), Image.shape[1],Image.shape[2], 1)

        return Image

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

    def normalization(self, image, nType='mean', reshape=True):

        NormalizedImage = []
        for im in image:
            if nType == 'mean':
                Imnormalize = (im - np.mean(im)) / np.std(im)

            if nType == -1:
                Imnormalize = 2 * (im - np.min(im)) / (
                        np.max(im) - np.min(im)) - 1

            if nType == 1:
                Imnormalize = (im - np.min(im)) / (
                        np.max(im) - np.min(im))

            if nType == 'RGB':
                Imnormalize = np.floor(im * 255)

            NormalizedImage.append(Imnormalize)

        NormalizedImage = np.array(NormalizedImage).astype('float')
        if reshape:
            if len(np.shape(NormalizedImage)) > 2:
                NormalizedImage = NormalizedImage.reshape(
                    len(NormalizedImage), NormalizedImage.shape[1],
                    NormalizedImage.shape[2], 1)
            else:
                NormalizedImage = NormalizedImage.reshape(1,
                                                          NormalizedImage.shape[
                                                              0],
                                                          NormalizedImage.shape[
                                                              1], 1)

        return NormalizedImage

    def addNoise(self, image):

        levelsNoise = np.arange(0.1, 1.0, 0.1)
        k = np.random.randint(0, len(levelsNoise))
        noise = np.random.normal(0.0, levelsNoise[k], image.shape)
        imageNoise = image + noise

        return imageNoise

    def applyTransform(self, image):
        from skimage.transform import rotate
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
        from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Activation,\
            Conv2DTranspose, Dropout
        from keras.models import Model

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
        from keras.models import Sequential, Model
        from keras.layers import Flatten, Dense, LeakyReLU, Input
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

    def train(self, X_train,epochs, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)
        self.true, self.noise = self.generate_data(X_train, 100)
        self.true = self.normalization(self.true, -1, False)
        self.noise = self.normalization(self.noise, -1, False)
        self.lossD = []
        self.lossG = []
        self.validation = []
        lossEpoch = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            imgs, noise1 = self.generate_data(X_train, half_batch)
            imgs = self.normalization(imgs,1, False)
            noise1 = self.normalization(noise1, -1, False)
            # Select a random half batch of images
	   
            # Generate a half batch of new images
            gen_imgs = self.predictionModel.predict(noise1)
            gen_imgs = self.normalization(gen_imgs,1, False)
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
            noise2 = self.normalization(noise2,-1,False)
            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise2, valid_y)

            # Plot the progress
            print ("%d/5000 [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss))

            evaluate = self.predictionModel.evaluate(self.noise, self.true)
            print "Validation =", evaluate

            if epoch > 500 and evaluate <= np.min(self.validation):
                self.predictionModelNoParallel.save(self.dir2)

            self.lossD.append(d_loss[0])
            self.lossG.append(g_loss)
            self.validation.append(evaluate)
            lossEpoch.append(d_loss[0])
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print "MeanLoss = ", np.mean(lossEpoch)
                self.save_imgs(epoch)
                lossEpoch = []

    def predict(self, model, data):

        test = data
        prediction = model.predict(test)

        predictEnhanced = self.normalization(prediction, 1)

        return predictEnhanced

    def save_imgs(self, epoch):
        gen_imgs = self.predictionModel.predict(self.noise)

        filename = "denoise_%d.png"
        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(10, 3)
        cnt = 0
        for i in range(10):
            axs[i, 0].imshow(self.true[cnt, :, :, 0], cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(self.noise[cnt, :, :, 0], cmap='gray')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, 2].axis('off')
            cnt += 1
        plt.savefig(filename % epoch)
        plt.close()




    
class UNET(GAN):

  def initModel(self, learningRate, boxSize, saveModelDir, numGPUs, init_n_filters=64):
    self.saveModelDir = saveModelDir
    self.numGPUs= numGPUs
    self.shape= (boxSize,boxSize,1)
    self.init_n_filters= init_n_filters
    self.batchSize= 16
    self.epochSize= 50 #In batches
    self.lr= learningRate
    
  def segmentationLoss(trn_labels_batch, logits):
    '''
    '''
    import tensorflow as tf
    logits=tf.reshape(logits, [-1])
    trn_labels=tf.reshape(trn_labels_batch, [-1])
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union))
    return loss


  def conv_block(self, m, dim, acti, bn, res, do=0):
      import keras
      n = keras.layers.Conv2D(dim, 3, activation=acti, padding='same')(m)
      n = keras.layers.BatchNormalization()(n) if bn else n
      n = keras.layers.Dropout(do)(n) if do else n
      n = keras.layers.Conv2D(dim, 3, activation=acti, padding='same')(n)
      n = keras.layers.BatchNormalization()(n) if bn else n
      return keras.layers.Concatenate()([m, n]) if res else n

  def level_block(self, m, dim, depth, inc, acti, do, bn, mp, up, res):
      import keras, math
      if depth > 0:
          n = self.conv_block(m, dim, acti, bn, res)
          m = keras.layers.MaxPooling2D()(n) if mp else keras.layers.Conv2D(dim, 3, strides=2, padding='same')(n)
          m = self.level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
          if up:
              m = keras.layers.UpSampling2D()(m)
              m = keras.layers.Conv2D(dim, 2, activation=acti, padding='same')(m)
          else:
              m = keras.layers.Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
          n = keras.layers.Concatenate()([n, m])
          m = self.conv_block(n, dim, acti, bn, res)
      else:
          m = self.conv_block(m, dim, acti, bn, res, do)
      return m

  def UNet(self, img_shape, out_ch=1, start_ch=32, depth=3, inc_rate=2., activation='relu', 
      dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
      import keras, math
      
      i = keras.layers.Input(shape=img_shape)
      PAD_SIZE= ( 2**(int(math.log(img_shape[1], 2))+1)-img_shape[1]) //2
      x = keras.layers.ZeroPadding2D( (PAD_SIZE, PAD_SIZE) )( i ) #Padded to 2**N
      o = self.level_block(x, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
      o = keras.layers.Conv2D(out_ch, 1, activation='linear')(o)
      o = keras.layers.Lambda(lambda m: m[:,PAD_SIZE:-PAD_SIZE,PAD_SIZE:-PAD_SIZE,:] )( o )
      return  keras.models.Model(inputs=i, outputs=o)
      

  def getDataGenerator(self, imgsMd, masksMd, augmentData=True, nEpochs=-1, isTrain=True, valFraction=0.1): 
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    import random, scipy
    def _random_flip_leftright( batchX, batchY):
      for i in range(batchX.shape[0]):
        if bool(random.getrandbits(1)):
          batchX[i] = np.fliplr(batchX[i])
          batchY[i] = np.fliplr(batchY[i])
      return batchX, batchY

    def _random_flip_updown( batchX, batchY):
      for i in range(batchX.shape[0]):
        if bool(random.getrandbits(1)):
          batchX[i] = np.flipud(batchX[i])
          batchY[i] = np.flipud(batchY[i])
      return batchX, batchY

    def _random_90degrees_rotation( batchX, batchY, rotations=[0, 1, 2, 3]):
      for i in range(batchX.shape[0]):
        num_rotations = random.choice(rotations)
        batchX[i] = np.rot90(batchX[i], num_rotations)
        batchY[i] = np.rot90(batchY[i], num_rotations)
      return batchX, batchY

    def _random_rotation( batchX, batchY, max_angle=25.):
      for i in range(batchX.shape[0]):
        if bool(random.getrandbits(1)):
          # Random angle
          angle = random.uniform(-max_angle, max_angle)
          batchX[i] = scipy.ndimage.interpolation.rotate(batchX[i], angle,reshape=False, mode="reflect")
          batchY[i] = scipy.ndimage.interpolation.rotate(batchY[i], angle,reshape=False, mode="reflect")
      return batchX, batchY

    def _random_blur( batchX, batchY, sigma_max):
      for i in range(batchX.shape[0]):
        if bool(random.getrandbits(1)):
          # Random sigma
          sigma = random.uniform(0., sigma_max)
          batchX[i] = scipy.ndimage.filters.gaussian_filter(batchX[i], sigma)
          batchY[i] = scipy.ndimage.filters.gaussian_filter(batchY[i], sigma)
      return batchX, batchY

    augmentFuns= [_random_flip_leftright, _random_flip_updown, _random_90degrees_rotation, _random_rotation]
    if augmentData:
      def augmentBatch( batchX, batchY):
        for fun in augmentFuns:
          if bool(random.getrandbits(1)):
            batchX, batchY= fun(batchX, batchY)
        return batchX, batchY
    else:
      def augmentBatch( batchX, batchY): return batchX, batchY


    if nEpochs<1:
      nEpochs= 9999999
    mdImgs  = md.MetaData(imgsMd)
    mdMasks  = md.MetaData(masksMd)  
    nImages= int(mdImgs.size())

    batchSize= self.batchSize
    stepsPerEpoch= nImages//batchSize
    I= xmippLib.Image()      

    imgFnames = mdImgs.getColumnValues(md.MDL_IMAGE)
    maskFnames= mdMasks.getColumnValues(md.MDL_IMAGE)
    (imgFnames_train, imgFnames_val, maskFnames_train,
     maskFnames_val) = train_test_split(imgFnames, maskFnames, test_size=valFraction, random_state=121)
    if isTrain:
      imgFnames, maskFnames= imgFnames_train, maskFnames_train
    else:
      imgFnames, maskFnames= imgFnames_val, maskFnames_val
      
    def dataIterator(imgFnames, maskFnames, nBatches=None):

      batchStack = np.zeros((batchSize,)+self.shape )
      batchLabels = np.zeros((batchSize,)+self.shape )
      currNBatches=0 
      for epoch in range(nEpochs):
        imgFnames, maskFnames= shuffle(imgFnames, maskFnames)
        n=0
        for fnImageImg, fnImageMask in zip(imgFnames, maskFnames):
          I.read(fnImageImg)
          batchStack[n,...]= np.expand_dims(I.getData(),-1)
          I.read(fnImageMask)
          batchLabels[n,...]= np.expand_dims(I.getData(),-1)
          n+=1
          if n>=batchSize:
            yield augmentBatch(batchStack, batchLabels)
            n=0
            currNBatches+=1
            if nBatches and currNBatches>=nBatches:
              break
        if n>0:
          yield augmentBatch(batchStack[:n,...], batchLabels[:n,...])
    return dataIterator(imgFnames, maskFnames), stepsPerEpoch
      
  def train(self, nEpochs, xmdParticles, xmdProjections):
  
    try:
      import keras
    except ImportError as e:
      print(e)
      raise ValueError(BAD_IMPORT_MSG)
      
    try:
      print("loading model ")
      model = keras.models.load_model(self.saveModelDir, custom_objects={})
      print("previous model loaded")
    except Exception as e:
      print(e)
      model = self.UNet( img_shape=self.shape, start_ch=self.init_n_filters, batchnorm=False )
      optimizer= keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0)
      model.compile(loss='mse', optimizer=optimizer)
      
      reduceLrCback= keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', min_lr=1e-8)
      earlyStopCBack= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
      saveModel= keras.callbacks.ModelCheckpoint(self.saveModelDir, monitor='val_loss', verbose=1, save_best_only=True)

      trainIterator, stepsPerEpoch= self.getDataGenerator(xmdParticles, xmdProjections, isTrain=True, valFraction=0.1)
      

      valIterator, stepsPerEpoch_val= self.getDataGenerator(xmdParticles, xmdProjections, augmentData=False, 
                                                             isTrain=False, valFraction=0.1)
      maxBatches= min(10, stepsPerEpoch_val)
      x_val=[]
      y_val=[]
      print("train/val split"); sys.stdout.flush()
      for i, (x, y) in enumerate(valIterator):
        x_val.append(x)
        y_val.append(y)
        if i== maxBatches:
          break
      valData=( np.concatenate(x_val, axis=0), np.concatenate(y_val, axis=0 ))
#      print(valData[0].shape)
      del valIterator
#      valData=None
      nEpochs_init= nEpochs
      nEpochs= max(1, nEpochs_init*float(stepsPerEpoch)/self.epochSize)
      print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
      sys.stdout.flush()
      
      model.fit_generator(trainIterator, epochs= nEpochs, steps_per_epoch=self.epochSize, #steps_per_epoch=stepsPerEpoch,
                        verbose=2, callbacks=[reduceLrCback, earlyStopCBack, saveModel],
                        validation_data=valData, max_queue_size=12, workers=1, use_multiprocessing=False)
                        
                        
