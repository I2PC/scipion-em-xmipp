import numpy as np
import pyworkflow.em.metadata as md
import xmippLib

BATCH_SIZE= 16
def getDataGenerator( imgsMd, masksMd, augmentData=True, nEpochs=-1, isTrain=True, valFraction=0.1, batchSize= BATCH_SIZE): 
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

  stepsPerEpoch= nImages//batchSize
  I= xmippLib.Image()      

  imgFnames = mdImgs.getColumnValues(md.MDL_IMAGE)
  maskFnames= mdMasks.getColumnValues(md.MDL_IMAGE)
  
  I.read( imgFnames[0] )
  shape= I.getData().shape+ (1,)
  if valFraction>0:
    (imgFnames_train, imgFnames_val, maskFnames_train,
     maskFnames_val) = train_test_split(imgFnames, maskFnames, test_size=valFraction, random_state=121)  
    if isTrain:
      imgFnames, maskFnames= imgFnames_train, maskFnames_train
    else:
      imgFnames, maskFnames= imgFnames_val, maskFnames_val
    
  def dataIterator(imgFnames, maskFnames, nBatches=None):
    
    batchStack = np.zeros((batchSize,)+shape )
    batchLabels = np.zeros((batchSize,)+shape )
    currNBatches=0 
    for epoch in range(nEpochs):
      if isTrain:
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
  
def extractNBatches(valIterator, maxBatches=-1):
  x_val=[]
  y_val=[]
  for i, (x, y) in enumerate(valIterator):
    x_val.append(x)
    y_val.append(y)
    if i== maxBatches:
      break
  return ( np.concatenate(x_val, axis=0), np.concatenate(y_val, axis=0 ))

def normalization( image, nType='mean', reshape=True):

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
