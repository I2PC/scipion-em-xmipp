# **************************************************************************
# *
# * Authors:     Javier Vargas (jvargas@cnb.csic.es)
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

from xmipp3.convert import writeSetOfParticles, xmippToLocation


class XmippProtCTFCorrectWiener2D(ProtProcessParticles):
    """    
     Performs CTF correction on images using Wiener filtering in 2D. This
     method enhances image quality by reducing noise and compensating for the
     contrast transfer function effects in the micrographs or particle images.
     Use it with caution, preferably only for visualization purposes or when
     subsequent methods demand it explicitly.

     AI Generated:

    ## Overview

    The CTF Correct Wiener 2D protocol performs a more complete correction of
    the contrast transfer function using **Wiener filtering**. Unlike simple
    phase flipping, this approach attempts to compensate not only for phase
    reversals but also for the attenuation of signal amplitudes, while
    simultaneously controlling the amplification of noise.

    In practical cryo-EM workflows, Wiener filtering is a more aggressive form
    of CTF correction. It can enhance the visual quality of particle images and
    partially restore high-frequency information that has been damped by the
    microscope. However, this comes at the cost of introducing assumptions
    about noise and signal that may not always hold.

    For biological users, this protocol should be understood as a **signal
    restoration technique** rather than a neutral preprocessing step. It is
    often useful for visualization or specific methods that require fully
    CTF-corrected images, but it must be applied with caution in standard
    reconstruction pipelines.

    ## Inputs and General Workflow

    The protocol takes as input a **set of particles** with associated CTF
    information. Using this information, it applies a Wiener filter to each
    image, producing a new set of particles where both phase and amplitude
    effects of the CTF have been corrected.

    If the input particles have already undergone phase flipping, the protocol
    detects this and takes it into account during processing, avoiding
    inconsistent corrections.

    The output is a new particle set with modified images that aim to
    approximate the true underlying signal more closely than the original data.

    ## Wiener Filtering: Conceptual Understanding

    The Wiener filter is designed to invert the effect of the CTF while
    controlling noise amplification. In regions where the CTF strongly
    attenuates the signal, a naive inversion would dramatically amplify noise.
    The Wiener approach balances this by incorporating a regularization term,
    preventing unstable corrections.

    From a biological perspective, this means that Wiener-filtered images may
    show enhanced contrast and clearer structural features. However, they are
    no longer raw experimental data—they are reconstructed estimates that
    depend on the assumed noise model.

    This is why Wiener correction is often considered more suitable for
    visualization or interpretation rather than as a standard input for all
    processing steps.

    ## Isotropic vs. Anisotropic Correction

    The protocol allows the user to perform an **isotropic correction**, which
    assumes that there is no astigmatism in the CTF. This simplifies the
    correction by treating the CTF as radially symmetric.

    This assumption is reasonable when astigmatism is negligible or when a
    simplified correction is sufficient. However, if astigmatism is
    significant, ignoring it may lead to suboptimal corrections, particularly
    at higher resolutions.

    In most practical situations, isotropic correction is acceptable for
    visualization purposes, but for more precise applications, users should
    consider whether astigmatism needs to be accounted for explicitly.

    ## Wiener Constant and Noise Control

    A key parameter in Wiener filtering is the **Wiener constant**, which
    controls the balance between signal restoration and noise suppression.

    Lower values favor stronger correction of the signal but risk amplifying
    noise. Higher values produce more conservative results, preserving
    stability at the expense of less aggressive correction.

    When the parameter is set to its default (negative value), the protocol
    uses a standard heuristic value. For most biological users, this default is
    appropriate unless there is a specific need to fine-tune the balance between
    noise and signal.

    ## Padding and Frequency Handling

    The **padding factor** determines how much the image is extended before
    applying the correction. Padding can improve the accuracy of Fourier-based
    operations by reducing edge effects.

    Increasing padding may lead to slightly better corrections, especially for
    high-resolution features, but also increases computational cost. In most
    cases, the default value provides a reasonable compromise.

    ## Envelope Correction

    The protocol optionally allows correction of the **CTF envelope**, which
    models additional damping effects such as beam coherence and specimen
    movement.

    This option should be used with caution. Envelope correction requires that
    the envelope function is accurately estimated. If this is not the case,
    applying the correction may introduce artifacts rather than improve the
    result.

    From a practical standpoint, this option is best reserved for
    well-characterized datasets where the envelope has been reliably modeled.

    ## Outputs and Their Interpretation

    The protocol produces a new set of particles in which both phase and
    amplitude effects of the CTF have been corrected using Wiener filtering.

    These particles often display:
    * Enhanced contrast
    * Improved visibility of structural features
    * Partial recovery of high-frequency information

    However, they should not be interpreted as raw experimental data. The
    correction introduces model-dependent modifications, and therefore the
    resulting images reflect both the data and the assumptions of the Wiener
    filter.

    ## Practical Recommendations

    In most cryo-EM workflows, Wiener-filtered particles are best used
    selectively. They are particularly useful for:
    * Visualization and inspection of particle quality
    * Generating illustrative figures
    * Methods that explicitly require Wiener-corrected inputs

    For standard reconstruction pipelines, many modern algorithms internally
    handle CTF effects in a statistically optimal way. In such cases, applying
    Wiener correction beforehand may not be necessary and can even interfere
    with the assumptions of those methods.

    A good strategy is to use this protocol as a complementary tool rather than
    a default preprocessing step.

    ## Final Perspective

    The CTF Correct Wiener 2D protocol provides a powerful but model-dependent
    way to restore image information affected by the microscope optics. By
    combining signal recovery with noise control, it can produce visually
    enhanced particle images that are easier to interpret.

    For biological users, the key is to understand its role: it is not simply
    correcting the data, but reconstructing an estimate of the underlying
    signal. When used appropriately, it can be highly informative, but it
    should always be applied with awareness of its assumptions and limitations.
    """
    _label = 'ctf_correct_wiener2d'
    
    def __init__(self, *args, **kwargs):
        ProtProcessParticles.__init__(self, *args, **kwargs)
        #self.stepsExecutionMode = STEPS_PARALLEL
        
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles', 
                      label="Input particles",  
                      help='Select the input projection images .') 
        form.addParam('isIsotropic', params.BooleanParam, default='True',
                      label="Isotropic Correction", 
                      help='If true, Consider that there is not astigmatism and then it is performed an isotropic correction.') 
        form.addParam('padding_factor', params.IntParam, default=2,expertLevel=params.LEVEL_ADVANCED,
                      label="Padding factor",  
                      help='Padding factor for Wiener correction ')
        form.addParam('wiener_constant', params.FloatParam, default=-1,expertLevel=params.LEVEL_ADVANCED,
                      label="Wiener constant",  
                      help=' Wiener-filter constant (if < 0: use FREALIGN default)')
        form.addParam('correctEnvelope', params.BooleanParam, default='False',expertLevel=params.LEVEL_ADVANCED,
                      label="Correct for CTF envelope",  
                      help=' Only in cases where the envelope is well estimated correct for it')                       
        form.addParallelSection(threads=1, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep',self.inputParticles.get().getObjId())        
        self._insertFunctionStep('wienerStep')
        self._insertFunctionStep('createOutputStep')
        
    def convertInputStep(self, particlesId):
        """ Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        writeSetOfParticles(self.inputParticles.get(), 
                            self._getPath('input_particles.xmd'))
    
    def wienerStep(self):
        params =  '  -i %s' % self._getPath('input_particles.xmd')
        params +=  '  -o %s' % self._getPath('corrected_ctf_particles.stk')
        params +=  '  --save_metadata_stack %s' % self._getPath('corrected_ctf_particles.xmd')
        params +=  '  --pad %s' % self.padding_factor.get()
        params +=  '  --wc %s' % self.wiener_constant.get()
        params +=  '  --sampling_rate %s' % self.inputParticles.get().getSamplingRate()

        if (self.inputParticles.get().isPhaseFlipped()):
            params +=  '  --phase_flipped '
            
        if (self.correctEnvelope):
            params +=  '  --correct_envelope '

        print(params)
        nproc = self.numberOfMpi.get()
        nT=self.numberOfThreads.get() 

        self.runJob('xmipp_ctf_correct_wiener2d', 
                    params, numberOfMpi=nproc,numberOfThreads=nT)
    
    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        imgFn = self._getPath('corrected_ctf_particles.xmd')
        
        partSet.copyInfo(imgSet)
        partSet.setIsPhaseFlipped(True)
        partSet.copyItems(imgSet,
                            updateItemCallback=self._updateLocation,
                            itemDataIterator=md.iterRows(imgFn, sortByLabel=md.MDL_ITEM_ID))
        
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)
    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        pass
    
    def _summary(self):
        pass
    
    def _methods(self):
        messages = []
        return messages
    
    def _citations(self):
        return []
    
    #--------------------------- UTILS functions -------------------------------------------- 
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)

