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
from xmipp3.base import XmippProtocol

IN_COORDS_POS_DIR_BASENAME= "pickNoiseInPosCoordinates"
OUT_COORDS_POS_DIR_BASENAME= "pickNoiseOutPosCoordinates"

class XmippProtPickNoise(ProtParticlePicking, XmippProtocol):
    """Protocol designed pick noise particles in micrographs and not real
    particles. The protocol allows you to choose the number of noise particles
    to extract from each micrograph. Set to -1 for extracting the same amount
    of noise particles as the number true particles for that micrograph.

    AI Generated

    ## Overview

    The Pick Noise protocol generates coordinates corresponding to background or
    noise regions in micrographs, rather than to true particles.

    The protocol starts from an existing set of particle coordinates. These input
    coordinates define where the real particles are expected to be. The protocol
    then selects additional coordinates in the same micrographs, but sufficiently
    far from the true particle positions, so that the selected boxes should mostly
    contain background noise.

    The output is a new set of coordinates representing noise particles. These
    coordinates can be used to extract noise images for training, validation,
    particle screening, method development, or comparison with real particle
    images.

    ## Inputs and General Workflow

    The main input is a set of true particle coordinates.

    The protocol first writes the input coordinates into Xmipp `.pos` files, one
    per micrograph. These files are used by the noise-picking program to know which
    regions should be avoided.

    The protocol then runs the Xmipp noise-picking tool on the directory containing
    the micrographs. For each micrograph, it selects coordinates away from the
    input particle coordinates. Finally, the generated noise coordinates are read
    back into Scipion as a new coordinate set.

    The output coordinate set is linked to the same micrographs as the input
    coordinate set.

    ## Input Coordinates

    The **Input coordinates** parameter should point to a SetOfCoordinates
    containing true particle positions.

    These coordinates define the exclusion regions for noise picking. In other
    words, the protocol uses them to avoid selecting boxes that overlap with known
    particles.

    The quality of the noise coordinates depends on the quality of the input
    particle coordinates. If the input set misses many particles, the protocol may
    accidentally select some real particles as noise. If the input set contains
    many false positives, valid background regions may be unnecessarily excluded.

    For this reason, it is preferable to use a reasonably clean particle-coordinate
    set as input.

    ## Number of Noise Particles

    The **Number of noise particles** parameter controls how many noise coordinates
    are selected from each micrograph.

    If a positive number is provided, the protocol tries to select that number of
    noise coordinates per micrograph.

    If the value is set to **-1**, the protocol selects the same number of noise
    coordinates as the number of true particle coordinates in each micrograph.

    The -1 option is useful when the user wants a balanced set of particle and
    noise examples. For example, if a micrograph contains 120 particle picks, the
    protocol will try to generate 120 noise picks for that micrograph.

    ## Box Size

    The protocol uses the box size stored in the input coordinate set.

    This box size defines the approximate region that would be extracted around
    each coordinate. It is important because noise coordinates should be far enough
    from true particles that the extracted noise boxes do not contain particle
    signal.

    If the box size is too small, the exclusion around particles may be too small.
    If it is too large, the protocol may reject too many possible background
    positions, especially in crowded micrographs.

    The output noise coordinate set keeps the corresponding box-size information.

    ## Noise Coordinates

    The coordinates produced by this protocol should be interpreted as background
    or non-particle positions.

    They are not guaranteed to contain pure noise in a strict physical sense.
    Depending on the micrograph, some boxes may include ice features, carbon,
    contamination, detector artifacts, or very weak unpicked particles. However,
    they are selected to avoid the known particle coordinates.

    This makes them useful as negative examples: image boxes that should not
    represent true particles according to the input coordinate set.

    ## Output Coordinates

    The main output is **outputCoordinates**, a SetOfCoordinates containing the
    picked noise positions.

    This output is associated with the same micrographs as the input coordinates.
    It can be passed to particle extraction protocols to extract noise boxes using
    the same extraction logic used for true particles.

    The output is especially useful when paired with the original particle
    coordinates. The user can extract both real particles and noise particles and
    compare them in later workflows.

    ## Typical Uses

    Noise coordinates can be useful in several situations.

    They can be used to train or evaluate particle-picking classifiers, where real
    particles are positive examples and noise boxes are negative examples.

    They can help test particle-screening methods by providing a known background
    class.

    They can be useful for estimating background statistics or for comparing
    particle images with non-particle image patches.

    They can also be used in method-development workflows where algorithms need
    examples of both signal and noise.

    ## Interpretation and Limitations

    The protocol assumes that regions far from known particle coordinates are
    reasonable noise candidates.

    This assumption is useful but not perfect. If the micrograph contains many
    unpicked particles, noise coordinates may include true particles. If the
    micrograph contains contamination or strong artifacts, some noise coordinates
    may correspond to structured non-particle features rather than simple
    background.

    Therefore, the output should be interpreted as “background/noise candidates”
    rather than a perfectly pure noise set.

    Visual inspection is recommended, especially if the noise coordinates will be
    used for training machine-learning models.

    ## Practical Recommendations

    Use a clean and representative input coordinate set. The protocol can only
    avoid particles that are present in the input coordinates.

    Use **-1** for the number of noise particles when you want approximately the
    same number of noise examples as real particle examples per micrograph.

    Use a fixed positive number when you want the same number of noise coordinates
    from every micrograph, regardless of particle density.

    Inspect extracted noise boxes before using them for training or validation.
    Remove micrographs with strong contamination if they produce misleading
    negative examples.

    Be careful with crowded micrographs. If particles are very dense, it may be
    difficult to find many background regions that are truly far from particles.

    Remember that the protocol creates coordinates only. To obtain actual noise
    images, run an extraction protocol using the generated noise coordinates.

    ## Final Perspective

    Pick Noise is a support protocol for generating negative examples from
    micrographs.

    For biological users, its main value is that it creates a coordinate set of
    background regions that can be processed in parallel with true particle
    coordinates. This is useful for particle-picking validation, classifier
    training, background analysis, and quality-control workflows.

    The protocol is most effective when the input particle coordinates are reliable
    and when the generated noise coordinates are visually checked before being used
    as negative examples.
    """
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



          
