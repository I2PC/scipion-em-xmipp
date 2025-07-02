# **************************************************************************
# *
# * Authors:     Federico P. de Isidro Gomez
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

import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pwem.protocols import ProtProcessParticles

from xmipp3.convert import writeSetOfParticles, xmippToLocation, readSetOfParticles


class XmippProtCTFCorrectPhase2D(ProtProcessParticles):
    """    
    Perform CTF correction by phase flip.
    """
    _label = 'ctf_correct_phase'
    
    def __init__(self, *args, **kwargs):
        ProtProcessParticles.__init__(self, *args, **kwargs)
        #self.stepsExecutionMode = STEPS_PARALLEL
        
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles', 
                      label="Input particles",  
                      help='Select the input projection images .')               
        form.addParallelSection(threads=1, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep',self.inputParticles.get().getObjId())        
        self._insertFunctionStep('phaseStep')
        self._insertFunctionStep('createOutputStep')
        
    def convertInputStep(self, particlesId):
        """ Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        writeSetOfParticles(self.inputParticles.get(), 
                            self._getPath('input_particles.xmd'))
    
    def phaseStep(self):
        params =  '  -i %s' % self._getPath('input_particles.xmd')
        params +=  '  -o %s' % self._getPath('corrected_ctf_particles.stk')
        params +=  '  --save_metadata_stack %s' % self._getPath('corrected_ctf_particles.xmd')
        params +=  '  --sampling_rate %s' % self.inputParticles.get().getSamplingRate()

        nproc = self.numberOfMpi.get()
        nT=self.numberOfThreads.get() 

        self.runJob('xmipp_ctf_correct_phase', 
                    params, numberOfMpi=nproc,numberOfThreads=nT)
    
    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        imgFn = self._getPath('corrected_ctf_particles.xmd')
        
        partSet.copyInfo(imgSet)
        partSet.setIsPhaseFlipped(True)
        readSetOfParticles(imgFn, partSet)
        
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)
