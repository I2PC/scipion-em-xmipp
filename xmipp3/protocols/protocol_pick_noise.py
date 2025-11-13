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

from pyworkflow import VERSION_2_0
from pyworkflow.object import Pointer

from pyworkflow.protocol.constants import (STEPS_SERIAL, LEVEL_ADVANCED)
import pyworkflow.protocol.params as params
from pwem.protocols import ProtParticlePicking

import pwem.emlib.metadata as MD
from xmipp3.convert import writeSetOfCoordinates, readSetOfCoordsFromPosFnames
from xmipp3 import XmippProtocol

IN_COORDS_POS_DIR_BASENAME= "pickNoiseInPosCoordinates"
OUT_COORDS_POS_DIR_BASENAME= "pickNoiseOutPosCoordinates"

class XmippProtPickNoise(ProtParticlePicking, XmippProtocol):
    """Protocol designed pick noise particles in micrographs and not real particles. The protocol allows you to choose the number of noise particles to extract from each micrograph. Set to -1 for extracting the same amount of noise particles as the number true particles for that micrograph """
    _label = 'pick noise'
    _lastUpdateVersion = VERSION_2_0
    
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
        inCoordsPosDir= os.path.join(outPath,  IN_COORDS_POS_DIR_BASENAME)
        outCoordsPosDir=os.path.join(outPath, OUT_COORDS_POS_DIR_BASENAME)
        
        mics= setOfCoords.getMicrographs()
        for mic in setOfCoords.getMicrographs():   
          mic_fname= mic.getFileName() 
          break
        mics_dir =  os.path.split(mic_fname)[0]
        argTuple=(boxSize, outPath, mics_dir, )
        deps=[]
        deps.append(self._insertFunctionStep("prepareInput"))
        deps.append(self._insertFunctionStep("pickNoiseStep", mics_dir, inCoordsPosDir, outCoordsPosDir, 
                                              boxSize, self.extractNoiseNumber.get(),
                                              self.numberOfThreads.get()) )
                                              
#        deps.append(self._insertFunctionStep("loadCoords", mics_dir, outCoordsPosDir, boxSize)
        
        self._insertFunctionStep('createOutputStep', outCoordsPosDir, prerequisites=deps)
        
    #--------------------------- STEPS functions -------------------------------
    
    def pickNoiseStep(self, mics_dir, inCoordsPosDir, outCoordsPosDir, 
                            boxSize, extractNoiseNumber, nThr):
                                              
        args=" -i %s -c %s -o %s -s %s -n %s -t %s"%(mics_dir, inCoordsPosDir, outCoordsPosDir, 
                                 boxSize, extractNoiseNumber,nThr)
                                 
        self.runJob('xmipp_pick_noise', args, numberOfMpi=1)
         

    def prepareInput(self):
      pickNoise_prepareInput(self.inputCoordinates.get(), self._getExtraPath())
        
    def createOutputStep( self, outCoordsPosDir):
      noiseCoords= readSetOfCoordsFromPosFnames(outCoordsPosDir, sqliteOutName=self._getExtraPath('consensus_NOISE.sqlite') ,
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
        else:
            summary.append("The output set of noisy particles has not finished yet")
        return summary



def pickNoise_prepareInput(setOfCoords, outPath, outCoordsPosDir=None): 

  boxSize = setOfCoords.getBoxSize()
  inCoordsFname= setOfCoords.getFileName()

  inCoordsPosDir= os.path.join(outPath,  IN_COORDS_POS_DIR_BASENAME)
  if not os.path.isdir(inCoordsPosDir):   os.mkdir(inCoordsPosDir) 
  
  if outCoordsPosDir is None:
    outCoordsPosDir=os.path.join(outPath, OUT_COORDS_POS_DIR_BASENAME)
  if not os.path.isdir(outCoordsPosDir):   os.mkdir(outCoordsPosDir)
  
  mics= setOfCoords.getMicrographs()
  for mic in setOfCoords.getMicrographs():   
    mic_fname= mic.getFileName() 
    break
  mics_dir =  os.path.split(mic_fname)[0]
  argTuple=(boxSize, outPath, mics_dir, )
        
  writeSetOfCoordinates(inCoordsPosDir, setOfCoords)

  return {"boxSize":boxSize, "inCoordsPosDir": inCoordsPosDir, "outCoordsPosDir":outCoordsPosDir,
          "outPath":outPath, "mics_dir":mics_dir }



          
