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

    AI Generated:

    ## Overview

    The CTF Correct Phase protocol applies a **phase-flipping correction** to a
    set of particle images. Its purpose is to compensate for the phase
    reversals introduced by the contrast transfer function (CTF) of the
    microscope, while leaving image amplitudes unchanged.

    In practical cryo-EM workflows, this is one of the simplest forms of CTF
    correction. It is typically used after particle extraction, once CTF
    parameters have already been assigned to the particles through their parent
    micrographs. The corrected particles can then be used in downstream steps
    such as classification, averaging, or reconstruction.

    For a biological user, the main idea is straightforward: some spatial
    frequencies in the experimental images have their sign inverted by the
    microscope optics, and phase flipping restores their correct sign. This
    often improves the internal consistency of the dataset and can make class
    averages and reconstructions easier to interpret.

    ## Inputs and General Workflow

    The protocol requires a **set of particles** as input. These particles must
    already carry valid CTF information, since the phase correction depends on
    the estimated CTF parameters associated with each particle.

    The protocol reads the input particles, applies the phase-flipping
    correction image by image, and writes a new set of corrected particles. The
    output therefore has the same particles as the input, but with their phases
    corrected according to the assigned CTF model.

    This is a direct preprocessing step: it does not classify, align, or
    reconstruct particles, but prepares them for later analysis.

    ## What Phase Flipping Does

    The contrast transfer function modulates the image formed in the microscope.
    One of its effects is to invert the sign of certain spatial frequencies. In
    other words, some structural information is transferred with the wrong phase.

    Phase flipping corrects this by reversing those sign inversions wherever
    the CTF predicts them. It is therefore a correction of the **phase** of the
    image signal, but not of the **amplitude** attenuation caused by the CTF.

    This distinction is important. Phase flipping is computationally simple and
    often beneficial, but it is not a complete CTF correction. High-frequency
    information that has been strongly damped by the microscope is not restored
    by this procedure alone.

    ## When to Use This Protocol

    This protocol is most useful in workflows where one wants a simple and
    robust CTF correction before downstream analysis. It is commonly used
    before 2D classification, 3D classification, or initial reconstruction,
    especially in pipelines where phase-flipped particles are expected by the
    following programs.

    From a biological point of view, phase flipping can improve the visibility
    of structural features in class averages and reduce the destructive effect
    of phase inversions on alignment and averaging.

    However, it is important to remember that this correction depends entirely
    on the quality of the CTF estimation. If the defocus or astigmatism
    parameters are inaccurate, the phase correction may be suboptimal and
    could even degrade the data.

    ## Requirements and Practical Considerations

    The essential requirement is that the input particles must have reliable
    CTF information associated with them. Usually, this means that CTF
    estimation has already been performed on the parent micrographs and that
    the particles were extracted in a way that preserves this metadata.

    A practical consequence is that this protocol should not be applied blindly
    to particles lacking validated CTF assignment. If the CTF estimation is
    poor, the resulting correction will also be poor.

    In addition, users should be aware that phase-flipped particles are a
    modified version of the original dataset. Some downstream methods assume
    raw particles, while others are fully compatible with phase-corrected
    inputs. It is therefore important to use this protocol consistently
    within the logic of the full workflow.

    ## Outputs and Their Interpretation

    The protocol produces a new **set of particles** in which the CTF phase
    inversions have been corrected. The metadata and general structure of the
    set are preserved, but the images themselves have been transformed.

    The output particles are explicitly marked as **phase flipped**, which is
    important for downstream processing and for keeping track of the history of
    the data.

    Biologically, the corrected particles should provide a cleaner basis for
    averaging and classification. Features that were partially canceled by
    inconsistent phase inversions may now combine more coherently across
    particles.

    ## Practical Recommendations

    In routine cryo-EM processing, this protocol is a reasonable choice when a
    simple and fast CTF correction is needed. It is particularly useful when
    preparing particles for methods that benefit from phase consistency but do
    not require a more elaborate amplitude correction.

    A good practice is to ensure that CTF estimation has already been carefully
    checked before running this step. If there are doubts about the CTF quality,
    it is better to address those earlier in the workflow rather than applying
    phase flipping to uncertain estimates.

    It is also advisable to keep track of whether downstream analyses are being
    performed on raw or phase-flipped particles, since mixing both kinds of
    inputs can lead to confusion in interpretation.

    ## Final Perspective

    The CTF Correct Phase protocol provides a simple and effective way to
    correct one of the main optical distortions introduced during cryo-EM image
    formation. Although it is not a full CTF restoration procedure, it often
    improves the coherence of particle data and supports more reliable
    downstream analysis.

    For most biological users, it should be viewed as a practical preprocessing
    step that helps align the experimental images more closely with the true
    underlying structure, provided that the underlying CTF estimation is
    trustworthy.

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
