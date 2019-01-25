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
from pyworkflow.protocol.constants import (STEPS_PARALLEL, STEPS_SERIAL, LEVEL_ADVANCED)
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
        self.stepsExecutionMode = STEPS_SERIAL

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      important=True,
                      label="Input coordinates",
                      help='Set of true particle coordinates. '
                           'Noise coordinates are chosen so that they are '
                           'sufficiently far from particles')
        
        form.addParam('extractNoiseNumber', params.IntParam,
                      default=-1, expertLevel=LEVEL_ADVANCED,
                      label='Number of noise particles',
                      help='Number of noise particles to extract from each micrograph. '
                           'Set to -1 for extracting the same amount of noise '
                           'particles as the number true particles for that micrograph')

        form.addParallelSection(threads=4, mpi=0)
    
    #--------------------------- INSERT steps functions ------------------------


    def _insertAllSteps(self):
        """for each micrograph insert the steps to preprocess it"""
        #Insert pickNoise steps using function

        setOfCoords= self.inputCoordinates.get()
        boxSize = setOfCoords.getBoxSize()
        inCoordsFname= setOfCoords.getFileName()
        outPath = self._getExtraPath()
        inCoordsPosDir= os.path.join(outPath,  "inPosCoordinates")
        outCoordsRawTxtDir=os.path.join(outPath, "outPosCoordinates")
        
        mics= setOfCoords.getMicrographs()
        for mic in setOfCoords.getMicrographs():   
          mic_fname= mic.getFileName() 
          break
        mics_dir =  os.path.split(mic_fname)[0]
        argTuple=(boxSize, outPath, mics_dir, )
        deps=[]
        deps.append(self._insertFunctionStep("prepareInput"))
        deps.append(self._insertFunctionStep("pickNoiseStep", mics_dir, inCoordsPosDir, outCoordsRawTxtDir, 
                                              boxSize, self.extractNoiseNumber.get(),
                                              self.numberOfThreads.get()) )
                                              
#        deps.append(self._insertFunctionStep("loadCoords", mics_dir, outCoordsRawTxtDir, boxSize)
        
        self._insertFunctionStep('createOutputStep', outCoordsRawTxtDir, prerequisites=deps)
        
    #--------------------------- STEPS functions -------------------------------
    
    def pickNoiseStep(self, mics_dir, inCoordsPosDir, outCoordsRawTxtDir, 
                            boxSize, extractNoiseNumber, nThr):
                                              
        args=" -i %s -c %s -o %s -s %s -n %s -t %s"%(mics_dir, inCoordsPosDir, outCoordsRawTxtDir, 
                                 boxSize, extractNoiseNumber,nThr)
                                 
        self.runJob('xmipp_pick_noise', args, numberOfMpi=1)
         

    def prepareInput(self):
      pickNoise_prepareInput(self.inputCoordinates.get(), self._getExtraPath())
        
    def createOutputStep( self, outCoordsRawTxtDir):
      noiseCoords= writeSetOfCoordsFromPosFnames(txt_coords_dirPath=outCoordsRawTxtDir, 
                                              setOfInputCoords=self.inputCoordinates.get())

      coordSet = self._createSetOfCoordinates(self.inputCoordinates.get().getMicrographs() )
      coordSet.copyInfo(noiseCoords)
      coordSet.copyItems(noiseCoords,
                         itemDataIterator=MD.iterRows(self._getPath(noiseCoords.getFileName()),
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

    def _summary(self):
        summary = []
        if hasattr(self, 'outputCoordinates'):
            summary.append('%d noisy particles were picked'%self.outputCoordinates.getSize())
        return summary


def writeSetOfCoordsFromPosFnames( txt_coords_dirPath, setOfInputCoords, sqliteOutName=None):
  '''
  txt_coords_dirPath: path where there is txt files with coordinates
  setOfInputCoords. Set to find micrographs
  setOfNoiseCoordinates if not none, set where results will be written.
  '''
  sufix="_raw_coords.txt"  
  inputMics = setOfInputCoords.getMicrographs()
  micIds= inputMics.getIdSet()
  micNameToMicId={}
  for micId in micIds:
    mic= inputMics[micId]
    micNameToMicId[".".join( mic.getMicName().split(".")[:-1] )]= micId

  if not sqliteOutName:
    sqliteOutName= os.path.join(txt_coords_dirPath, 'consensus_NOISE.sqlite')
  cleanPath('coordinates_randomPick.sqlite')
  cleanPath(sqliteOutName)

  setOfNoiseCoordinates= SetOfCoordinates(filename= sqliteOutName)
  setOfNoiseCoordinates.setMicrographs(inputMics)
  setOfNoiseCoordinates.setBoxSize( setOfInputCoords.getBoxSize())

  for fname in os.listdir( txt_coords_dirPath ):
    if sufix is None or fname.endswith(sufix):
      with open(os.path.join(txt_coords_dirPath, fname)) as f:
        mic_name= f.readline().split()[0]
        for line in f:
          x_new, y_new = line.split()
          x_new, y_new = int(x_new), int(y_new)
          try:
            aux = Coordinate()
            aux.setMicrograph( inputMics[micNameToMicId[mic_name]] )
            aux.setX(x_new)
            aux.setY(y_new)
            setOfNoiseCoordinates.append(aux )
          except KeyError as e:
            print( e)
            continue

  setOfNoiseCoordinates.write()
  return setOfNoiseCoordinates


def pickNoise_prepareInput(setOfCoords, outPath, outCoordsRawTxtDir=None): 

  boxSize = setOfCoords.getBoxSize()
  inCoordsFname= setOfCoords.getFileName()

  inCoordsPosDir= os.path.join(outPath,  "inPosCoordinates")
  if outCoordsRawTxtDir is None:
    outCoordsRawTxtDir=os.path.join(outPath, "outPosCoordinates")
  
  mics= setOfCoords.getMicrographs()
  for mic in setOfCoords.getMicrographs():   
    mic_fname= mic.getFileName() 
    break
  mics_dir =  os.path.split(mic_fname)[0]
  argTuple=(boxSize, outPath, mics_dir, )
        
  inCoordsFname= setOfCoords.getFileName()
  inCoordsPosDir= os.path.join(outPath,  "inPosCoordinates")
  if not os.path.isdir(inCoordsPosDir):   os.mkdir(inCoordsPosDir) 
  outCoordsRawTxtDir=os.path.join(outPath, "outPosCoordinates")
  if not os.path.isdir(outCoordsRawTxtDir):   os.mkdir(outCoordsRawTxtDir)
  writeSetOfCoordinates(inCoordsPosDir, setOfCoords, 
                          scale=setOfCoords.getBoxSize())

  return {"boxSize":boxSize, "inCoordsPosDir": inCoordsPosDir, "outCoordsRawTxtDir":outCoordsRawTxtDir,
          "outPath":outPath, "mics_dir":mics_dir }


#def loadCoords(mics_dir, outCoordsRawTxtDir, boxSize):
#    setOfCoordinates = SetOfCoordinates(filename=self._getExtraPath("picknoise_coords.sqlite") )
#    setOfCoordinates.setMicrographs( self.inputCoordinates.get() )
#    setOfCoordinates.setBoxSize(boxSize)
#        
#    for micrograph in inputMics:
#        fnTmp = os.path.join(outCoordsRawTxtDir, '%s_raw_coords.txt' % micrograph.getObjId())
#        if os.path.exists(fnTmp):
#            coords = np.loadtxt(fnTmp, skiprows=1)
#            if coords.size == 2:  # special case with only one coordinate in consensus
#                coords = [coords]
#            for coord in coords:
#                aux = Coordinate()
#                aux.setMicrograph(micrograph)
#                aux.setX(coord[0])
#                aux.setY(coord[1])
#                setOfCoordinates.append(aux)
#    setOfCoordinates.write()
