
from .protocol_generate_reprojections import XmippProtGenerateReprojections
import pyworkflow.protocol.params as params
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D, \
    BatchNormalization, LeakyReLU, Flatten, Activation, Dense, \
    Conv2DTranspose, MaxPooling2D, Dropout
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
    return -K.mean((2. * intersection + 1) / (union + 1), axis=0)


class XmippProtDeepDenoising(XmippProtGenerateReprojections):

    _label ="deep denoising"

    def _defineParams(self, form):

        form.addSection('Input')
        form.addParam('deepMsg', params.LabelParam, default=True,
                      label='WARNING! You need to have installed '
                            'Keras programs')
        form.addParam('model', params.EnumParam, choices=['Train & Predict',
                                                          'Predict'],
                       default=ITER_TRAIN,
                       label='Train or predict model',
                       help='*Train*: Train the model using noisy particles '
                            'or their projections in an initial volume'
                            '*Predict*: The particles are denoised with a '
                            'pretrained model')
        form.addParam('inputProjections', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      condition='model==%d'%ITER_TRAIN,

                      label='Input projections of the volume to train',
                      help='use '
                                                                       'the '
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

        if 'CUDA' in os.environ and not os.environ['CUDA'] == "False":

            form.addParam('gpuToUse', params.IntParam, default='0',
                          label='Which GPU to use:',
                          help='Currently just one GPU will be use, by '
                               'default GPU number 0 You can override the default '
                               'allocation by providing other GPU number, p.e. 2'
                               '\nif -1 no gpu but all cpu cores will be used')

        form.addParallelSection(threads=1, mpi=8)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.newXdim = self.imageSize.get()
        self._insertFunctionStep('preprocessData')
        if self.model.get() == ITER_TRAIN:
            self._insertFunctionStep('trainModel')
        self._insertFunctionStep('predictModel')
        self._insertFunctionStep('createOutputStep')

    def preprocessData(self):
        particles = self._getExtraPath('noisyParticles.xmd')
        writeSetOfParticles(self.inputParticles.get(), particles)
        self.metadata = xmippLib.MetaData(particles)
        fnNewParticles = self._getExtraPath('resizedParticles.stk')
        self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
            particles, fnNewParticles, self.newXdim))

        if self.model.get() == ITER_TRAIN:
            projections = self._getExtraPath('projections.xmd')
            writeSetOfParticles(self.inputProjections.get(), projections)
            fnNewProjections = self._getExtraPath('resizedProjections.stk')
            self.runJob("xmipp_image_resize", "-i %s -o %s --dim %d" % (
                projections, fnNewProjections, self.newXdim))

        dir = self._getPath('ModelTrained.h5')
        self.gan = GAN()
        self.gan.setSize(self.imageSize.get())
        self.gan.initModel(dir)


    def trainModel(self):
        self.X_train = self.gan.extractTrainData(self._getExtraPath(
            'resizedProjections.xmd'), xmippLib.MDL_IMAGE, -1)

        self.gan.train(self.X_train, epochs=5000,
                                       batch_size=32, save_interval=500)

        self.Generator = load_model(self._getPath('ModelTrained.h5'))

    def predictModel(self):
        metadataPart = xmippLib.MetaData(self._getExtraPath(
            'resizedParticles.xmd'))
        if self.model.get() == ITER_TRAIN:
            metadataProj = xmippLib.MetaData(self._getExtraPath(
            'resizedProjections.xmd'))
        img = xmippLib.Image()
        dimMetadata = getMdSize(self._getExtraPath(
                'resizedParticles.xmd'))
        xmippLib.createEmptyFile(self._getExtraPath(
            'particlesDenoised.stk'), self.newXdim, self.newXdim, 1,
            dimMetadata)

        mdNewParticles = md.MetaData()

        self.groupParticles = 5000

        if self.model.get() == ITER_PREDICT:
            if self.modelPretrain:
                '''model = load_model(self.ownModel.get(), custom_objects={
                    'dice_coef': dice_coef})'''
                model = load_model(self.ownModel.get())
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
        self.runJob("xmipp_transform_normalize", "-i %s --method NewXmipp "
                                                 "--background circle %d "%
                    (self._getExtraPath('particlesDenoised.stk'),
                     self.newXdim/2))

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
        if self.model.get() == ITER_TRAIN:
            setXmippAttributes(particle, row,
                               xmippLib.MDL_CORR_DENOISE_PROJECTION,
                               xmippLib.MDL_CORR_DENOISE_NOISY)
        else:
            setXmippAttributes(particle, row, xmippLib.MDL_CORR_DENOISE_NOISY)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Particles denoised")
        return summary


class GAN(XmippProtDeepDenoising):

    def initModel(self, saveModelDir):

        self.dir2 = saveModelDir
        optimizer = Adam(0.0001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error',
                               optimizer=optimizer)

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

    def setSize(self, size):

        self.img_rows = size
        self.img_cols = size
        self.channels = 1
        self.shape = self.img_rows * self.img_cols
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

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

        levelsNoise = np.arange(0.1, 1.5, 0.1)
        k = np.random.randint(0, len(levelsNoise))
        noise = np.random.normal(0.0, levelsNoise[k], image.shape)
        imageNoise = image + noise

        return imageNoise

    def applyTransform(self, image):

        angle = np.random.randint(-180, 180)
        #shiftx = np.random.randint(-10, 10)
        #shifty = np.random.randint(-10, 10)

        imRotate = rotate(image, angle, mode='wrap')
        #shift = AffineTransform(translation=[shiftx, shifty])
        #imShift = warp(imRotate, shift, mode='wrap',
        #               preserve_range=True)

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
        x = Conv2D(64, (9, 9), padding='same')(input_img)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (5, 5), padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        encoded = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2DTranspose(64, kernel_size=1, strides=1, padding='same')(
            encoded)
        x = BatchNormalization(momentum=0.8)(x)
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
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        '''model.add(Conv2D(32 * 1, 5, strides=2, input_shape=img_shape, \
                          padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(32 * 2, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Conv2D(32 * 4, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        # Out: 1-dim probability
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))'''

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)
        self.true, self.noise = self.generate_data(X_train, 100)
        self.lossD = []
        self.lossG = []
        self.validation = []
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
            print ("%d/5000 [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss))

            evaluate = self.generator.evaluate(self.noise, self.true)
            print "Validation =", evaluate

            if epoch > 0 and evaluate <= np.min(self.validation):
                self.generator.save(self.dir2)

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
        gen_imgs = self.generator.predict(self.noise)

        filename = "denoise_%d.png"
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

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

