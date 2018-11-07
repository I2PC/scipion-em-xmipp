# **************************************************************************
# *
# * Authors:     Ruben Sanchez
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

import os
from pyworkflow.object import Pointer
import pyworkflow.utils as pwutils
from pyworkflow.utils.path import cleanPath
from pyworkflow.protocol.constants import (STEPS_PARALLEL, LEVEL_ADVANCED)
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtParticlePicking
from pyworkflow.em.data import SetOfCoordinates, Coordinate
import pyworkflow.em.metadata as MD
from xmipp3.convert import writeSetOfCoordinates, readSetOfCoordinates
from xmipp3 import XmippProtocol
import numpy as np
from scipy.spatial.distance import cdist
from pyworkflow.object import Float

class XmippProtPickNoise(ProtParticlePicking, XmippProtocol):
    """Protocol to pick noise particles"""
    _label = 'pick noise'
    
    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      important=True,
                      label="Input coordinates",
                      help='Set of true particle coordinates. Noise coordinates are chosen'
                           'so that they are sufficiently far from particles')
        
        form.addParam('extractNoiseNumber', params.IntParam, default=-1, expertLevel=LEVEL_ADVANCED,
                      label='Number of noise particles',
                      help='Number of noise particles to extract from each micrograph. '
                           'Set to -1 for extracting the same amount of noise particles as the number true particles for that micrograph')

        form.addParallelSection(threads=4, mpi=1)
    
    #--------------------------- INSERT steps functions ------------------------
    
    def pickNoiseWorker(self, fun, *args):
        return fun(*args)
    
    def _insertAllSteps(self):
        """for each micrograph insert the steps to preprocess it"""
        #Insert pickNoise steps using function

        func, argList= pickAllNoiseWorker( self.inputCoordinates.get(), self._getExtraPath(), self.extractNoiseNumber.get() )
        deps=[]
        for argTuple in argList:
            deps.append(self._insertFunctionStep("pickNoiseWorker", func, *argTuple) )
        
        self._insertFunctionStep('createOutputStep', prerequisites= deps )


    #--------------------------- STEPS functions -------------------------------
    def writePosFilesStep(self):
        """ Write the pos file for each micrograph on metadata format. """
        writePosFilesStepWorker(self.inputCoordinates.get(), self._getExtraPath())

    def pickNoiseStep(self, mic_id, mic_fname, mic_shape, outputRoot, extractNoiseNumber, boxSize, coords_in_mic_list):
        pickNoiseOneMic( mic_id, mic_fname, mic_shape, outputRoot, extractNoiseNumber, boxSize, coords_in_mic_list)      
        
    def createOutputStep( self):
      noiseCoords= writeSetOfCoordsFromFnames(txt_coords_dirPath=self._getExtraPath(), 
                                          setOfInputCoords=self.inputCoordinates.get())

      coordSet = self._createSetOfCoordinates(self.inputCoordinates.get().getMicrographs() )
      coordSet.copyInfo(noiseCoords)
      coordSet.copyItems(noiseCoords, itemDataIterator=MD.iterRows(self._getPath(noiseCoords.getFileName()),
                                                                   sortByLabel=MD.MDL_ITEM_ID))
      coordSet.setBoxSize(noiseCoords.getBoxSize())      
      self._defineOutputs(outputCoordinates=coordSet)
      self._defineSourceRelation(self.inputCoordinates.get(), coordSet)
      
      
    def getInputMicrographs(self):
        return self.inputCoordinates.get().getMicrographs()
    
    def getInputMicrographsPointer(self):
        ptr = Pointer()
        ptr.set(self.getInputMicrographs())
        return ptr
    
    def readSetOfCoordinates(self, workingDir, coordSet):
        readSetOfCoordinates(workingDir, self.getInputMicrographs(), coordSet)

    def _summary(self):
        summary = []
        if hasattr(self, 'outputCoordinates'):
            summary.append('%d noisy particles were picked'%self.outputCoordinates.getSize())
        return summary


def writePosFilesStepWorker(setOfCoords, outpath):
    """ Write the pos file for each micrograph on metadata format. """
    writeSetOfCoordinates(outpath, setOfCoords, 
                          scale=setOfCoords.getBoxSize())
  
      

def pickAllNoiseWorker( setOfCoords, outputRoot, numberOfParticlesToPick):
    '''
    return pickNoiseFunction, listOfArgumentsTo pickNoiseFunction
    '''

    argsList=[]
    boxSize = setOfCoords.getBoxSize()
    for mic in setOfCoords.getMicrographs():   
        mic_fname= mic.getFileName() 
        mic_id= mic.getObjId()
        mic_shape= mic.getDim()
        coords_in_mic_list= list(setOfCoords.iterCoordinates(mic.getObjId() ) )     
        argsList.append( (mic_id, mic_fname, mic_shape, outputRoot, 
                          numberOfParticlesToPick, boxSize, coords_in_mic_list) )
    return pickNoiseOneMic, argsList


def writeSetOfCoordsFromFnames( txt_coords_dirPath, setOfInputCoords, sqliteOutName=None):
  '''
  txt_coords_dirPath: path where there is txt files with coordinates
  setOfInputCoords. Set to find micrographs
  setOfNoiseCoordinates if not none, set where results will be written.
  '''
  sufix="_raw_coords.txt"  
  inputMics = setOfInputCoords.getMicrographs()
  if not sqliteOutName:
    sqliteOutName= os.path.join(txt_coords_dirPath, 'consensus_NOISE.sqlite')
  cleanPath('coordinates_randomPick.sqlite')
  setOfNoiseCoordinates= SetOfCoordinates(filename= sqliteOutName)
  setOfNoiseCoordinates.setMicrographs(inputMics)
  setOfNoiseCoordinates.setBoxSize( setOfInputCoords.getBoxSize())

  for fname in os.listdir( txt_coords_dirPath ):
    if sufix is None or fname.endswith(sufix):
      with open(os.path.join(txt_coords_dirPath, fname)) as f:
        mic_id= int(f.readline().split()[0])
        mic= inputMics[mic_id]
        for line in f:
          x_new, y_new = line.split()
          x_new, y_new = int(x_new), int(y_new)
          aux = Coordinate()
          aux.setMicrograph(mic)
          aux.setX(x_new)
          aux.setY(y_new)
          setOfNoiseCoordinates.append(aux)
  setOfNoiseCoordinates.write()
  return setOfNoiseCoordinates

def pickNoiseOneMic( mic_id, mic_fname, mic_shape, outputRoot, extractNoiseNumber, boxSize, coords_in_mic_list):
    """ Pick noise from one micrograph 
        protocol is self or any other pyworkflow.em.protocol 
    """
    extractNoiseNumber= extractNoiseNumber if extractNoiseNumber>0 else len(coords_in_mic_list)
    baseMicName = pwutils.removeBaseExt(mic_fname)
    min_required_distance= boxSize//2
    currentCoords= []
    for coord in coords_in_mic_list:
      currentCoords.append( (coord.getX(), coord.getY()) )
    currentCoords= np.array(currentCoords)
    if currentCoords.shape[0]>0:
      good_new_coords= []
      n_good_coords= 0
      for iterNum in range(9999):
        randomCoordsX= np.random.randint( boxSize//2, mic_shape[0]- boxSize//2 , size=extractNoiseNumber*2 )
        randomCoordsY= np.random.randint( boxSize//2, mic_shape[1]- boxSize//2 , size=extractNoiseNumber*2 )
        randomCoords= np.stack( [randomCoordsX, randomCoordsY], axis=1)
        del randomCoordsX, randomCoordsY
        dist_mat= cdist(randomCoords, currentCoords) #dist_mat: i:random_particles   j: currentCoords
        dist_mat= np.min(dist_mat, axis=1)
        g_n_c= randomCoords[ dist_mat>= min_required_distance ]
        n_good_coords+= g_n_c.shape[0]
        good_new_coords.append(g_n_c )
        if n_good_coords>= extractNoiseNumber:
          break
      good_new_coords= np.concatenate(good_new_coords)[:extractNoiseNumber]    
      with open(os.path.join(outputRoot, baseMicName+"_raw_coords.txt"), "w") as f:
          f.write("%d %s %s\n"%(mic_id, baseMicName, mic_fname))
          for x, y in good_new_coords:
            f.write("%d %d\n"%(x,y))
 
