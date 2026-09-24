# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
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

from pwem.objects import Volume
from pwem.protocols import ProtReconstruct3D
from pwem import emlib
from pwem.emlib.metadata import getFirstRow
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pyworkflow.utils.path import createLink
from xmipp3.convert import writeSetOfParticles
from xmipp3.base import isXmippCudaPresent
import os
from pyworkflow import UPDATED, PROD

class XmippProtReconstructFourier(ProtReconstruct3D):
    """    
    Reconstruct a volume using Xmipp_reconstruct_fourier from a given set of
    particles.
    The alignment parameters will be converted to a Xmipp xmd file
    and used as direction projections to reconstruct.

    AI Generated

    ## Overview

    The Reconstruct Fourier protocol reconstructs a 3D volume from a set of
    particles with known projection-alignment parameters.

    In single-particle cryo-EM, each particle image is interpreted as a 2D
    projection of the 3D structure from a particular direction. Once particles
    have projection angles and shifts, those images can be inserted into Fourier
    space and combined to produce a 3D map. This protocol performs that
    reconstruction using Xmipp Fourier reconstruction programs.

    The protocol can optionally correct the CTF before reconstruction, impose
    symmetry, limit the maximum resolution used in Fourier space, and generate
    independent half-map reconstructions for resolution assessment.

    The main output is a reconstructed volume. If half maps are requested, the
    output volume also keeps the two half-map file names associated with it.

    ## Inputs and General Workflow

    The main input is a set of particles with projection-alignment information.

    The protocol first converts the input particle set into Xmipp metadata format.
    If CTF information is available and CTF correction is requested, the particles
    are corrected by Wiener filtering before reconstruction. Otherwise, the
    original particle metadata are used directly.

    If the **Use halves** option is enabled, the corrected particle metadata are
    split into two random subsets. Each subset is reconstructed independently, and
    the two resulting half maps are averaged to produce the final output volume.

    If halves are not requested, all particles are reconstructed together into a
    single output volume.

    Finally, the reconstructed volume is registered in Scipion with the sampling
    rate of the input particles.

    ## Input Particles

    The **Input particles** parameter should point to a SetOfParticles with 3D
    projection alignment.

    This is essential. The protocol does not determine particle orientations from
    scratch. It assumes that each particle already has projection angles and shifts
    from a previous refinement, angular assignment, classification, or imported
    alignment.

    If the angular assignments are poor, the reconstruction will also be poor. The
    output volume should therefore be interpreted in relation to the quality of the
    input particle alignment.

    The input particles should also have a consistent box size, sampling rate, and
    contrast convention.

    ## Symmetry Group

    The **Symmetry group** parameter defines the symmetry imposed during
    reconstruction.

    For asymmetric particles, use **c1**. If the particle has known point-group
    symmetry, the corresponding Xmipp symmetry group can be specified.

    Correct symmetry can improve the reconstruction by averaging equivalent views
    and increasing signal. However, incorrect symmetry can introduce artificial
    density, blur asymmetric features, or obscure real biological differences.

    Users should impose symmetry only when it is justified by prior structural or
    biological knowledge.

    ## Maximum Resolution

    The **Maximum resolution** parameter limits the highest-resolution information
    used during Fourier reconstruction.

    The value is given in angstroms. If it is set to -1, the protocol uses the
    Nyquist limit.

    Limiting the maximum resolution can be useful when the input particles or
    angular assignments are not reliable at high frequency. It can reduce the
    influence of high-frequency noise during reconstruction.

    Internally, the angstrom value is converted to a digital frequency using the
    particle sampling rate. The reconstruction program then uses this value as the
    maximum Fourier-space resolution.

    ## CTF Correction

    The **Correct CTF** option applies Wiener-filter CTF correction to the
    particles before reconstruction, when CTF metadata are available.

    CTF correction compensates for the contrast transfer effects introduced by the
    microscope. This can make the reconstructed map more physically meaningful,
    especially when particles come from different defocus values.

    If the particles already have CTF information, the protocol writes a corrected
    stack and metadata file before reconstruction. If the particles are marked as
    phase-flipped, this information is passed to the CTF-correction step.

    If CTF information is not present, the protocol simply reconstructs from the
    input particle metadata without CTF correction.

    ## Correct CTF Envelope

    The **Correct CTF envelope** option is available when CTF correction is
    enabled.

    The CTF envelope models additional attenuation of signal at higher spatial
    frequencies. Correcting it may be useful when the envelope has been estimated
    reliably.

    This option should be used with caution. If the envelope estimate is poor,
    correcting it may amplify noise or introduce artifacts.

    Most users should enable it only when they understand how the envelope was
    estimated and why it is appropriate for the dataset.

    ## Use Halves

    The **Use halves** option creates two independent reconstructions from two
    random subsets of the input particles.

    This is useful for resolution estimation and validation. Half maps are commonly
    used to compute FSC curves and to assess reproducibility of structural signal.

    When this option is enabled, the protocol:

    1. splits the particle metadata into two subsets;
    2. reconstructs half map 1;
    3. reconstructs half map 2;
    4. averages the two half maps to create the final output volume;
    5. stores the half-map file names in the output volume metadata.

    The final averaged map can be used for visualization or downstream processing,
    while the half maps can be used for validation.

    ## Padding Factor

    The **Padding factor** parameters control padding of the input projections and
    the reconstructed volume during Fourier reconstruction.

    There are two values:

    - **Projection padding**;
    - **Volume padding**.

    Padding can improve interpolation accuracy in Fourier space, but it increases
    memory use and computation time. Larger padding values may therefore be more
    accurate but slower and more demanding.

    The default values are a practical compromise for many datasets. Advanced users
    may adjust them when reconstruction accuracy or performance needs to be tuned.

    ## Approximative Version

    The **Approximative version** option enables a faster approximation of the
    Fourier reconstruction algorithm.

    When enabled, reconstruction is faster but may be slightly less precise than
    the full version.

    This option is useful for routine reconstruction or exploratory workflows where
    speed is important. If maximum numerical precision is required, advanced users
    may disable the approximation.

    The approximative version is not compatible with the legacy CPU implementation.

    ## Legacy Version

    The **Legacy version** option uses the original CPU implementation of the
    Fourier reconstruction algorithm.

    This option is provided mainly for backward compatibility. In routine use, it
    should usually not be necessary.

    The legacy version cannot be used with GPU execution and is not compatible with
    the approximative version.

    ## Extra Parameters

    The **Extra parameters** field allows advanced users to pass additional options
    to the underlying Xmipp Fourier reconstruction program.

    This can be useful for specialized workflows that require options not exposed
    directly in the graphical form.

    Most users should leave this field empty. Incorrect extra parameters may cause
    the reconstruction to fail or produce unexpected results.

    ## GPU and CPU Execution

    The protocol supports GPU and CPU execution.

    GPU execution is enabled by default and uses the Xmipp CUDA Fourier
    reconstruction program when available. GPU execution is usually faster and is
    recommended for large datasets.

    If GPU execution is requested but the required Xmipp CUDA programs are not
    available, the protocol reports a validation error.

    The CPU version has limitations. In particular, the non-GPU version can use
    only a single thread; MPI should be used instead for CPU parallelism. The
    legacy version is CPU-only.

    ## Output Volume

    The main output is **outputVolume**.

    This volume is reconstructed from the input particles and registered with the
    same sampling rate as the input particle set.

    If **Use halves** is disabled, the output volume is reconstructed directly from
    all particles.

    If **Use halves** is enabled, the output volume is the average of the two
    independent half-map reconstructions, and the half-map file names are stored
    with the output volume.

    The output volume can be used for visualization, post-processing, FSC
    calculation, refinement assessment, or as input to other 3D analysis
    protocols.

    ## Interpreting the Reconstruction

    The reconstructed volume reflects the input particles and their assigned
    orientations.

    Good input alignments, appropriate CTF handling, and correct symmetry usually
    lead to a more interpretable map. Poor alignments, incorrect CTF metadata,
    wrong symmetry, or inconsistent particle populations can produce blurred or
    distorted density.

    If half maps are generated, they should be used for validation. Agreement
    between half maps provides evidence that the reconstructed features are
    supported reproducibly by independent subsets of particles.

    ## Practical Recommendations

    Use this protocol after particles have reliable 3D projection alignment.

    Enable CTF correction when reliable CTF metadata are available and the
    downstream reconstruction strategy requires corrected particles.

    Use half maps when you plan to estimate resolution or validate the map with
    FSC.

    Use symmetry only when it is biologically justified.

    Keep the default padding values unless memory use, speed, or reconstruction
    accuracy requires adjustment.

    Use GPU execution when available. It is usually the practical choice for large
    particle sets.

    Inspect the output volume and, when generated, compare the two half maps. Poor
    agreement between half maps may indicate overfitting, bad alignments,
    heterogeneity, or insufficient particle quality.

    ## Final Perspective

    Reconstruct Fourier is a core 3D reconstruction protocol. It converts a set of
    projection-aligned particles into a 3D map using Fourier-space reconstruction.

    For biological users, the key point is that the protocol does not solve the
    orientation problem; it uses orientations that already exist in the input
    particles. Therefore, the quality of the reconstructed volume depends strongly
    on the quality of the previous alignment or refinement step.

    Used with reliable particle alignments, appropriate CTF correction, and
    well-chosen symmetry, the protocol produces maps and half maps that can support
    subsequent validation, post-processing, interpretation, and refinement.
    """
    _label = 'reconstruct fourier'
    _devStatus = PROD

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):

        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')

        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles', pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help='Select the input images from the project.')     
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group", 
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp') 
        form.addParam('maxRes', params.FloatParam, default=-1,
                      label="Maximum resolution (A)",  
                      help='Maximum resolution (in Angstrom) to consider \n'
                           'in Fourier space (default Nyquist).\n'
                           'Param *--maxres* in Xmipp.')
        form.addParam('useHalves', params.BooleanParam, label='Use halves', default=False,
                      help='Create separate reconstructions from two random subsets. Useful for resolution measurements')
        form.addParam('correctCTF', params.BooleanParam, label='Correct CTF', default=True,
                      help='Correct the CTF with a Wiener filter before reconstructing, if the CTF is available')
        form.addParam('correctEnvelope', params.BooleanParam, label='Correct CTF envelope', default=False, condition="correctCTF",
                      help='Correct the CTF envelope')
        line = form.addLine('Padding factor',
                             expertLevel=cons.LEVEL_ADVANCED,
                             help='Padding of the input images. Higher number will result in more precise interpolation in Fourier '
                             'domain, but slower processing time and higher memory requirements.')
        line.addParam('pad_proj', params.IntParam, default=2, label='Projection')
        line.addParam('pad_vol', params.IntParam, default=2, label='Volume')

        form.addParam('legacy', params.BooleanParam, default=False,
                      label="Legacy version",
                      expertLevel=cons.LEVEL_ADVANCED,
                      help="Use original CPU version of the algorithm. This should not be necessary, but it's present"
                           " to ensure backward compatibility")

        form.addParam('approx', params.BooleanParam, default=True,
                      label="Approximative version",
                      expertLevel=cons.LEVEL_ADVANCED,
                      help="If on, an approximation of the original algorithm will be used. This will result in"
                           " faster processing times, but (slightly) less precise result")

        form.addParam('extraParams', params.StringParam, default='', expertLevel=cons.LEVEL_ADVANCED,
                      label='Extra parameters: ', 
                      help='Extra parameters to *xmipp_(cuda_)reconstruct_fourier* program:\n'
                      """
                      --iter () : Subtract projections of this map from the images used for reconstruction
                      """)

        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called for iterations and references. """
        myDict = {
            'input_xmd': self._getTmpPath('input_particles.xmd'),
            'input_tmp_root': self._getTmpPath('input_particles_corrected'),
            'half1_xmd': self._getTmpPath('input_particles_corrected000001.xmd'),
            'half2_xmd': self._getTmpPath('input_particles_corrected000002.xmd'),
            'output_volume': self._getPath('output_volume.mrc'),
            'half1_volume': self._getPath('half1.mrc'),
            'half2_volume': self._getPath('half2.mrc')
            }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep('convertInputStep')
        if self.useHalves.get():
            self._insertFunctionStep('splitInputStep')
            self._insertReconstructStep('half1')
            self._insertReconstructStep('half2')
            self._insertFunctionStep('averageStep')
        else:
            self._insertReconstructStep()
        self._insertFunctionStep('createOutputStep')
        
    def _insertReconstructStep(self, half=None):
        if half is None:
            params =  '  -i %s.xmd' % self._getFileName('input_tmp_root')
            params += '  -o %s' % self._getFileName('output_volume')
        else:
            params =  '  -i %s' % self._getFileName(half + '_xmd')
            params += '  -o %s' % self._getFileName(half + '_volume')
            
        params += ' --sym %s' % self.symmetryGroup.get()
        maxRes = self.maxRes.get()
        if maxRes == -1:
            digRes = 0.5
        else:
            digRes = self.inputParticles.get().getSamplingRate() / self.maxRes.get()
        params += ' --max_resolution %0.3f' %digRes
        params += ' --padding %0.3f %0.3f' % (self.pad_proj.get(), self.pad_vol.get())
        params += ' --sampling %f' % self.inputParticles.get().getSamplingRate()
        params += ' %s' % self.extraParams.get()
        params += ' --fast' if self.approx.get() else ''

        if self.useGpu.get():
            #AJ to make it work with and without queue system
            params += ' --thr %d' % self.numberOfThreads.get()
            if self.numberOfMpi.get()>1:
                N_GPUs = len((self.gpuList.get()).split(','))
                params += ' -gpusPerNode %d' % N_GPUs
                params += ' -threadsPerGPU %d' % max(self.numberOfThreads.get(),4)
            count=0
            GpuListCuda=''
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux
            if self.numberOfMpi.get()==1:
                params += ' --device %s'%(GpuListCuda) if self.useGpu.get() else ''

        self._insertFunctionStep('reconstructStep', params)
        
    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        particlesMd = self._getFileName('input_xmd')
        imgSet = self.inputParticles.get()
        writeSetOfParticles(imgSet, particlesMd)
        fnCorrectedImagesRoot = self._getFileName('input_tmp_root')

        row = getFirstRow(particlesMd)
        hasCTF = row.containsLabel(emlib.MDL_CTF_DEFOCUSU) or row.containsLabel(emlib.MDL_CTF_MODEL)
        if hasCTF and self.correctCTF:
            args = "-i %s -o %s.stk --save_metadata_stack %s.xmd --keep_input_columns" %\
                   (particlesMd, fnCorrectedImagesRoot, fnCorrectedImagesRoot)
            args += " --sampling_rate %f" % imgSet.getSamplingRate()
            if self.correctEnvelope:
                args+=" --correct_envelope"
            if imgSet.isPhaseFlipped():
                args += " --phase_flipped"
            Nproc = self.numberOfThreads.get() * self.numberOfMpi.get()
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=min(Nproc, 24))
            self.runJob("xmipp_image_eliminate_byEnergy", "-i %s.xmd --sigma2 9 --minSigma2 0.01" % \
                        fnCorrectedImagesRoot, numberOfMpi=min(Nproc, 12))
        else:
            createLink(particlesMd,fnCorrectedImagesRoot+".xmd")

    def splitInputStep(self):
        args = ['-i', self._getFileName('input_tmp_root')+".xmd"]
        args += ['-n', 2]

        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
        
    def reconstructStep(self, params):
        """ Create the input file in STAR format as expected by Xmipp.
        If the input particles comes from Xmipp, just link the file. 
        """
        if self.useGpu.get():
            if self.numberOfMpi.get()>1:
                self.runJob('xmipp_cuda_reconstruct_fourier', params, numberOfMpi=len((self.gpuList.get()).split(','))+1)
            else:
                self.runJob('xmipp_cuda_reconstruct_fourier', params)
        else:
            if self.legacy.get():
                self.runJob('xmipp_reconstruct_fourier', params)
            else:
                self.runJob('xmipp_reconstruct_fourier_accel', params)
            
    def averageStep(self):
        # Read
        half1 = emlib.Image(self._getFileName('half1_volume'))
        half2 = emlib.Image(self._getFileName('half2_volume'))
        
        # Average
        half1.inplaceAdd(half2)
        half1.inplaceMultiply(0.5)
        
        # Write
        half1.write(self._getFileName('output_volume'))
        
    def createOutputStep(self):
        imgSet = self.inputParticles.get()

        self.runJob("xmipp_image_header", "-i %s --sampling_rate %f"%\
                    (self._getFileName('output_volume'), imgSet.getSamplingRate()),
                    numberOfMpi=1)
        
        volume = Volume()
        volume.setFileName(self._getFileName('output_volume'))
        volume.setSamplingRate(imgSet.getSamplingRate())
        if self.useHalves.get():
            volume.setHalfMaps([
                self._getFileName('half1_volume'),
                self._getFileName('half2_volume')
            ])
        
        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles, volume)
    
    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        errors = ProtReconstruct3D._validate(self)
        if self.useGpu.get() and self.legacy.get():
            errors.append("Legacy version is not implemented for GPU")
        if self.approx.get() and self.legacy.get():
            errors.append("Approximative version is not implemented for Legacy code")
        if not self.useGpu.get() and self.numberOfThreads.get() > 1:
            errors.append("CPU version can use only a single thread. Use MPI instead")
        if self.useGpu and not isXmippCudaPresent():
            errors.append("You have asked to use GPU, but I cannot find Xmipp GPU programs in the path")
        return errors
    
    def _summary(self):
        """ Should be overriden in subclasses to 
        return summary message for NORMAL EXECUTION. 
        """
        return []
    
    #--------------------------- UTILS functions --------------------------------------------
