import sys, os
import numpy as np
from keras.models import load_model
import mrcfile # scipion run pip install mrcfile

from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.io import imsave, imread
from skimage import util 
from scipy.ndimage import gaussian_filter

BATCH_SIZE=8

def normalize(img):
  normData= (img -np.min(img))/ (np.max(img)-np.min(img))
  normData= 2*normData -1
  return normData
    
def main():
  defaultModelName="/home/rsanchez/app/xmipp_bundle/src/scipion-em-xmipp/xmipp3/protocols/deepDenoisingWorkers/denoiseMicModel/modelTrainiedDenoiseMic_px_57.h5"
  patch_size= int(os.path.basename(defaultModelName).split(".")[0].split("_")[-1]) #Default is 57
  model= load_model(defaultModelName)
  oneMic= loadMic("/home/rsanchez/ScipionUserData/rawData/10005/mics/stack_0001_2x_SumCorr.mrc")
  print( np.mean(oneMic), np.max(oneMic), np.min(oneMic))
  oneMic= util.invert(oneMic)
  print( np.mean(oneMic), np.max(oneMic), np.min(oneMic))
#  plt.hist(oneMic[:])
#  plt.show()
#  plotSeveralMics([oneMic], ["original"])
  print("data loaded")
  
  predictedMic= predictMicro(model, oneMic, patch_size)

  plotSeveralMics([oneMic, predictedMic], ["original", "denoised"])
  
def loadMic(fnameIn):
  if fnameIn.split(".")[-1].startswith("mrc"):
    with  mrcfile.open(fnameIn, permissive=True) as mrc:
      micData= np.squeeze( mrc.data)
  else:
    micData= np.squeeze( imread(fnameIn))
  return  micData
  
def predictMicro(model, micro, patch_size):
  microHight= micro.shape[0]
  microWidth= micro.shape[1]
  micro_list=[]
  halfPatch= patch_size//2
  halfPatchRemainder= patch_size%2
  startPoint= halfPatch
  endPoint_width= microWidth - patch_size//2
  endPoint_hight= microHight - patch_size//2
  for i in range(startPoint, endPoint_hight, patch_size):
    sys.stdout.write("\r%d"%(i)); sys.stdout.flush()
    for j in range(startPoint, endPoint_width, patch_size):
      image= micro[i-halfPatch: i+halfPatch+halfPatchRemainder, j-halfPatch: j+halfPatch+halfPatchRemainder].reshape(1, patch_size, patch_size, 1)
      micro_list.append(normalize(image))
  micro_list= np.concatenate(micro_list) #.reshape((-1, microHight, microWidth, 1))
  micro_list_pred= model.predict(micro_list, batch_size=BATCH_SIZE)
#  micro_list_pred= micro_list

  micro_pred= np.zeros((microHight, microWidth))
  k=0
  for i in range(startPoint, endPoint_hight, patch_size):
    for j in range(startPoint, endPoint_width, patch_size):
      micro_pred[i-halfPatch: i+halfPatch+halfPatchRemainder, j-halfPatch: j+halfPatch+halfPatchRemainder]= micro_list_pred[k,:,:,0]
      k+=1
  return micro_pred

def plotSeveralMics(listOfMicsAsNp, listOfNames, fiterSigmaFactor= 0.002, mainTitle=None):
  fig= plt.figure()
  if mainTitle:
    fig.suptitle(mainTitle)  
  nFigs= len(listOfMicsAsNp)
  for i in range(1, nFigs+1):
    ax = fig.add_subplot(1, nFigs, i)
    ax.set_title(listOfNames[i-1])
    micAsNp= gaussian_filter(listOfMicsAsNp[i-1], int(fiterSigmaFactor*listOfMicsAsNp[i-1].shape[0]))
    plt.imshow(micAsNp, cmap="Greys")
  plt.show()
  
if __name__=="__main__":
  main()
  
