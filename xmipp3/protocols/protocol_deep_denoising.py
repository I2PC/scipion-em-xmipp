
from .protocol_generate_reprojections import XmippProtGenerateReprojections
import pyworkflow.protocol.params as params
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D, \
    BatchNormalization, LeakyReLU, Flatten, Activation, Dense
from keras import backend as K
import xmippLib
from pyworkflow.em.data import Image
from skimage.transform import rotate, AffineTransform, warp
from xmipp3.convert import writeSetOfParticles, setXmippAttributes, xmippToLocation
import matplotlib.pyplot as plt
from xmipp3.utils import isMdEmpty, getMdSize
from xmipp3.convert import mdToCTFModel, readCTFModel
import pyworkflow.em.metadata as md
from scipy.stats import pearsonr
from pyworkflow.utils.path import cleanPath
import os

ITER_TRAIN = 0
ITER_PREDICT = 1


def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + 1) / (union + 1), axis=0)


class XmippProtDeepDenoising(XmippProtGenerateReprojections):

    _label ="deep denoising"

    def _defineParams(self, form):

        form.addSection('Input')
        form.addParam('deepMsg', params.LabelParam, default=True,
                      label='WARNING! You need to have installed '
                            'Keras programs')
        form.addParam('model', params.EnumParam, choices=['Train',
                                                           'Predict'],
                       display=params.EnumParam.DISPLAY_HLIST,
                       default=ITER_TRAIN,
                       label='Train or predict model',
                       help='*Train*: Train the model using noisy particles '
                            'or their projections in an initial volume'
                            '*Predict*: The particles are denoised with a '
                            'pretrained model')
        form.addParam('inputProjections', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      condition='model==%d'%ITER_TRAIN,
                      label='Input projections of the volume', help='use the '
                      'protocol generate reprojections to generate the '
                      'reprojections views')

        form.addParam('modelPretrain', params.BooleanParam, default = False,
                      condition='model==%d'%ITER_PREDICT,label='Choose your '
                      'own model', help='Setting "yes" '
                      'you can choose your own model trained. If you choose'
                      '"no" a general model pretrained will be assign')

        form.addParam('ownModel', params.PathParam,
                      condition='modelPretrain==True and model==%d'%ITER_PREDICT,
                      label='Set your model',
                      help='Set your model trained in format .h5')

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      label='Input noisy particles', help='Input noisy '
                      'particles from the protocol generate reprojections if '
                      'you are training or from any other protocol if you are '
                      'predicting')

        if 'CUDA' in os.environ and not os.environ['CUDA'] == "False":

            form.addParam('gpuToUse', params.IntParam, default='0',
                          label='Which GPU to use:',
                          help='Currently just one GPU will be use, by '
                               'default GPU number 0 You can override the default '
                               'allocation by providing other GPU number, p.e. 2'
                               '\nif -1 no gpu but all cpu cores will be used')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('preprocessData')
        if self.model.get() == ITER_TRAIN:
            self._insertFunctionStep('trainModel')
        self._insertFunctionStep('predict')
        self._insertFunctionStep('createOutputStep')

    def preprocessData(self):

        particles = self._getExtraPath('noisyParticles.xmd')
        writeSetOfParticles(self.inputParticles.get(), particles)
        fnNewParticles = self._getExtraPath('resizedParticles.stk')
        newXdim = 100
        self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
            particles, fnNewParticles, newXdim))

        if self.model.get() == ITER_TRAIN:
            projections = self._getExtraPath('projections.xmd')
            writeSetOfParticles(self.inputProjections.get(), projections)
            fnNewProjections = self._getExtraPath('resizedProjections.stk')
            self.runJob("xmipp_image_resize", "-i %s -o %s --dim %d" % (
                projections, fnNewProjections, newXdim))

        self.gan = GAN()

    def trainModel(self):
        self.X_train = self.gan.extractTrainData(self._getExtraPath(
            'resizedProjections.xmd'), xmippLib.MDL_IMAGE, -1)

        self.Generator = self.gan.train(self.X_train, epochs=10000,
                                       batch_size=32, save_interval=500)

        self.Generator.save(self._getPath('ModelTrained.h5'))

    def predict(self):
        metadataPart = xmippLib.MetaData(self._getExtraPath(
            'resizedParticles.xmd'))
        if self.model.get() == ITER_TRAIN:
            metadataProj = xmippLib.MetaData(self._getExtraPath(
            'resizedProjections.xmd'))
        img = xmippLib.Image()
        dimMetadata = getMdSize(self._getExtraPath(
                'resizedParticles.xmd'))
        xmippLib.createEmptyFile(self._getExtraPath(
            'particlesDenoised.stk'), 100, 100, 1,
            dimMetadata)

        mdNewParticles = md.MetaData()

        self.groupParticles = 2000

        if self.model.get() == ITER_PREDICT:
            if self.modelPretrain:
                model = load_model(self.ownModel.get(), custom_objects={
                    'dice_coef': dice_coef})
            else:
                model = load_model(self._getPath('PretrainModel.h5'))
        for num in range(1,dimMetadata,self.groupParticles):
            self.noisyParticles = self.gan.extractInfoMetadata(metadataPart,
                                    xmippLib.MDL_IMAGE, img, num, self.groupParticles, 2)

            if self.model.get() == ITER_TRAIN:
                self.predict = self.gan.predict(self.Generator, self.noisyParticles)
                self.projections = self.gan.extractInfoMetadata(metadataProj,
                                    xmippLib.MDL_IMAGE, img, num, self.groupParticles, 1)
            else:
                self.predict = self.gan.predict(model, self.noisyParticles)

            self.noisyParticles = self.gan.normalization(self.noisyParticles, 1)

            self.prepareMetadata(mdNewParticles, img, num)

        mdNewParticles.write('particles@' + self._getExtraPath(
            'particlesDenoised.xmd'), xmippLib.MD_APPEND)

    def prepareMetadata(self, metadata, image, initItem):
        newRow = md.Row()
        for i, img in enumerate(self.predict):
            correlations2, _ = pearsonr(img.ravel(), self.noisyParticles[
                i].ravel())
            image.setData(np.squeeze(img))
            path = '%06d@' % (i + initItem) + self._getExtraPath(
                   'particlesDenoised.stk')
            image.write(path)

            pathNoise = '%06d@' % (i + initItem) + self._getExtraPath(
                'resizedParticles.stk')

            newRow.setValue(md.MDL_IMAGE, path)
            newRow.setValue(md.MDL_IMAGE_ORIGINAL, pathNoise)
            if self.model.get() == ITER_TRAIN:
                pathProj = '%06d@' % (i + initItem) + self._getExtraPath(
                    'resizedProjections.stk')
                newRow.setValue(md.MDL_IMAGE_REF, pathProj)
                correlations1, _ = pearsonr(img.ravel(), self.projections[
                    i].ravel())
                newRow.setValue(md.MDL_CORR_DENOISE_PROJECTION, correlations1)

            newRow.setValue(md.MDL_CORR_DENOISE_NOISY, correlations2)
            newRow.addToMd(metadata)


    def createOutputStep(self):

        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(self.inputParticles.get())
        imgFn = self._getExtraPath('particlesDenoised.xmd')
        self.iterMd = md.iterRows(imgFn, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(self.inputParticles.get(),
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
        setXmippAttributes(particle, row,
                           xmippLib.MDL_CORR_DENOISE_PROJECTION,
                           xmippLib.MDL_CORR_DENOISE_NOISY)
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Particles denoised")
        return summary


class GAN():
    def __init__(self):
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 1
        self.shape = self.img_rows * self.img_cols
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=dice_coef,
                               optimizer=optimizer)
        'mean_squared_error'

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.img_shape)  # Input(shape=(self.shape,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
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

    def normalization(self, image, type='mean', reshape=True):

        NormalizedImage = []
        for im in image:
            if type == 'mean':
                Imnormalize = (im - np.mean(im)) / np.std(im)

            if type == -1:
                Imnormalize = 2 * (im - np.min(im)) / (
                        np.max(im) - np.min(im)) - 1

            if type == 1:
                Imnormalize = (im - np.min(im)) / (
                        np.max(im) - np.min(im))

            if type == 'RGB':
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

        levelsNoise = np.arange(0.5, 2.5, 0.1)
        k = np.random.randint(0, len(levelsNoise))
        noise = np.random.normal(0.0, levelsNoise[k], image.shape)
        imageNoise = image + noise

        return imageNoise

    def applyTransform(self, image):

        angle = np.random.randint(-180, 180)
        shiftx = np.random.randint(-10, 10)
        shifty = np.random.randint(-10, 10)

        imRotate = rotate(image, angle, mode='wrap')
        shift = AffineTransform(translation=[shiftx, shifty])
        imShift = warp(imRotate, shift, mode='wrap',
                       preserve_range=True)

        return imShift

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
        x = Conv2D(32, (15, 15), padding='same')(input_img)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        encoded = AveragePooling2D((2, 2), padding='same',
                                   name='encoder')(x)
        x = Conv2D(64, (3, 3), padding='same')(encoded)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2D(2)(x)
        x = Conv2D(1, (5, 5), padding='same')(x)
        decoded = Activation('linear')(x)

        model = Model(input_img, decoded)

        noise = Input(shape=self.img_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)
        self.true, self.noise = self.generate_data(X_train, 100)
        self.lossD = []
        self.lossG = []
        lossEpoch = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            imgs, noise1 = self.generate_data(X_train, half_batch)
            # Select a random half batch of images

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise1)
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

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise2, valid_y)

            # Plot the progress
            print ("%d/10000 [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss))

            self.lossD.append(d_loss[0])
            self.lossG.append(g_loss)
            lossEpoch.append(d_loss[0])
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print "MeanLoss = ", np.mean(lossEpoch)
                self.save_imgs(X_train, epoch)
                lossEpoch = []

        return self.generator

    def predict(self, model, data, evaluate=False):

        test = data
        prediction = model.predict(test)

        NoiseEnhanced = []
        for i, im in enumerate(prediction):
            NewNoise = (im + data[i]) / 2.0
            NoiseEnhanced.append(NewNoise)

        NoiseEnhanced = np.asarray(NoiseEnhanced).astype('float32')
        NoiseEnhanced = NoiseEnhanced.reshape(len(NoiseEnhanced),
                                              NoiseEnhanced.shape[1],
                                              NoiseEnhanced.shape[2], 1)
        predictEnhanced = model.predict(NoiseEnhanced)
        predictEnhanced = self.normalization(predictEnhanced, 1)

        return predictEnhanced

    def save_imgs(self, X_train, epoch):
        gen_imgs = self.generator.evaluate(self.noise, self.true)
        print "Validation =", gen_imgs

