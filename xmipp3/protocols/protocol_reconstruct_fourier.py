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
    Reconstruct a volume using Xmipp_reconstruct_fourier from a given set of particles.
    The alignment parameters will be converted to a Xmipp xmd file
    and used as direction projections to reconstruct.
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
