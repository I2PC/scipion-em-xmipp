# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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

import math
import os
from glob import glob
from shutil import copy

from pyworkflow.utils import Timer, join
from pyworkflow.utils.path import cleanPattern, cleanPath, makePath, moveFile
from pyworkflow.protocol.params import *

import pwem.emlib.metadata as metadata
from pwem.protocols import ProtInitialVolume
from pwem.constants import ALIGN_NONE
from pwem.objects import SetOfClasses2D, Volume
from pwem import emlib

from xmipp3.constants import CUDA_ALIGN_SIGNIFICANT
from xmipp3.convert import writeSetOfClasses2D, writeSetOfParticles, volumeToRow
from xmipp3.base import isXmippCudaPresent


class XmippProtReconstructSignificant(ProtInitialVolume):
    """
    This algorithm addresses the initial volume problem in SPA
    by setting it in a Weighted Least Squares framework and
    calculating the weights through a statistical approach based on
    the cumulative density function of different image similarity measures.
    """
    _label = 'reconstruct significant'

    # --------------------------- DEFINE param functions -----------------------

    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input classes",
                      important=True,
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Select the input classes2D from the project.\n'
                           'It should be a SetOfClasses2D class with  class '
                           'representative')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]'
                           'for a description of the symmetry groups format.'
                           ' If no symmetry is present, give c1.')
        form.addParam('thereisRefVolume', BooleanParam, default=False,
                      label="Is there a reference volume(s)?",
                      help='You may use a reference volume to initialize  '
                           'the calculations. For instance, this is very  '
                           'useful to obtain asymmetric volumes from '
                           'symmetric  references. The symmetric reference  '
                           'is provided as starting point, choose no '
                           'symmetry  group (c1), and reconstruct_significant'
                           'will tend to break the symmetry finding a '
                           'suitable  volume. The reference volume can also be '
                           'useful, for instance, when reconstructing a '
                           'fiber.  Provide in this case a cylinder of a  '
                           'suitable size.')
        form.addParam('refVolume', PointerParam,
                      label='Initial 3D reference volumes',
                      pointerClass='Volume', condition="thereisRefVolume")
        form.addParam('angularSampling', FloatParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Angular sampling',
                      help='Angular sampling in degrees for generating the '
                           'projection gallery.')
        form.addParam('minTilt', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      label='Minimum tilt (deg)',
                      help='Use the minimum and maximum tilts to limit the  '
                           'angular search. This can be useful, for instance, '
                           'in the reconstruction of fibers from side views. '
                           '0 degrees is a top view, while 90 degrees is a  '
                           'side view.')
        form.addParam('maxTilt', FloatParam, default=180,
                      expertLevel=LEVEL_ADVANCED,
                      label='Maximum tilt (deg)',
                      help='Use the minimum and maximum tilts to limit the  '
                           'angular search. This can be useful, for instance, '
                           'in the reconstruction of fibers from side views. '
                           '0 degrees is a top view, while 90 degrees is a  '
                           'side view.')
        form.addParam('maximumShift', FloatParam, default=-1,
                      expertLevel=LEVEL_ADVANCED,
                      label='Maximum shift (px):',
                      help="Set to -1 for free shift search")
        form.addParam('keepIntermediate', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
                      label='Keep intermediate volumes',
                      help='Keep all volumes and angular assignments along  '
                           'iterations')
        form.addParam('useMaxRes', BooleanParam, default=False,
                      label="Use new maximum resolution?",
                      help='You may use a new maximum resolution to simplify '
                           'the calculations keeping only low frequency '
                           'information.',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('maxResolution', FloatParam,
                      label="Target resolution", default=12,
                      help='Target resolution (A).', condition='useMaxRes',
                      expertLevel=LEVEL_ADVANCED)

        form.addSection(label='Criteria')
        form.addParam('alpha0', FloatParam, default=80,
                      label='Starting significance',
                      help='80 means 80% of significance. Use larger numbers '
                           'to relax the starting significance and have a  '
                           'smoother landscape of solutions')
        form.addParam('iter', IntParam, default=50,
                      label='Number of iterations',
                      help='Number of iterations to go from the initial  '
                           'significance to the final one')
        form.addParam('alphaF', FloatParam, default=99.5,
                      label='Final significance',
                      help='99.5 means 99.5% of significance. Use smaller  '
                           'numbers to be more strict and have a sharper  '
                           'reconstruction. Be aware that if you are too '
                           'strict, you may end with very few projections  '
                           'and the reconstruction becomes very'
                           'noisy.')
        form.addParam('useImed', BooleanParam, default=True,
                      expertLevel=LEVEL_ADVANCED,
                      label='Use IMED',
                      help='Use IMED for the weighting. IMED is an '
                           'alternative to correlation that can discriminate '
                           'better among very similar images')
        form.addParam('strictDir', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
                      label='Strict direction',
                      help='If the direction  is strict, then only the most  '
                           'significant experimental images can contribute  '
                           'to it. As a consequence, many experimental '
                           'classes are lost and only the best contribute '
                           'to the 3D reconstruction. Be aware that only the '
                           'best can be very few depending on the cases.')
        form.addParam('angDistance', IntParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Angular neighborhood',
                      help='Images in an angular neighborhood also determines '
                           'the weight of each image. It should be at least  '
                           'twice the angular sampling')
        form.addParam('dontApplyFisher', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
                      label='Do not apply Fisher',
                      help="Images are preselected using Fisher's confidence "
                           "interval on the correlation coefficient. "
                           "Check this box if you do not want to make "
                           "this preselection.")

        form.addParallelSection(threads=1, mpi=8)

    # --------------------------- INSERT steps functions -----------------------

    def getSignificantArgs(self, imgsFn):
        """ Return the arguments needed to launch the program. """
        # Prepare arguments to call program
        self._params = {'imgsFn': imgsFn,
                        'extraDir': self._getExtraPath(),
                        'symmetryGroup': self.symmetryGroup.get(),
                        'angularSampling': self.angularSampling.get(),
                        'minTilt': self.minTilt.get(),
                        'maxTilt': self.maxTilt.get(),
                        'maximumShift': self.maximumShift.get(),
                        'angDistance': self.angDistance.get()
                        }
        args = '-i %(imgsFn)s --sym %(symmetryGroup)s --angularSampling ' \
               '%(angularSampling)f --minTilt %(minTilt)f --maxTilt ' \
               '%(maxTilt)f ' '--maxShift %(maximumShift)f --dontReconstruct ' \
               '--angDistance %(angDistance)f' % self._params

        if self.useImed:
            args += " --useImed"
        if self.strictDir:
            args += " --strictDirection"
        if self.dontApplyFisher:
            args += " --dontApplyFisher"

        return args

    def _insertAllSteps(self):
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_classes.xmd')
        self._insertFunctionStep('convertInputStep', self.imgsFn)
        SL = emlib.SymList()
        SL.readSymmetryFile(self.symmetryGroup.get())
        self.trueSymsNo = SL.getTrueSymsNo()
        self.TsCurrent = self.inputSet.get().getSamplingRate()

        n = self.iter.get()
        alpha0 = self.alpha0.get()
        deltaAlpha = (self.alphaF.get() - alpha0) / n

        # Insert one step per iteration
        for i in range(n):
            alpha = 1 - (alpha0 + deltaAlpha * i) / 100.0
            self._insertFunctionStep('significantStep', i + 1, alpha)

        self._insertFunctionStep('createOutputStep')

        # --------------------------- STEPS functions --------------------------

    def significantStep(self, iterNumber, alpha):
        iterDir = self._getTmpPath('iter%03d' % iterNumber)
        makePath(iterDir)
        prevVolFn = self.getIterVolume(iterNumber - 1)
        volFn = self.getIterVolume(iterNumber)
        anglesFn = self._getExtraPath('angles_iter%03d.xmd' % iterNumber)

        t = Timer()
        t.tic()
        if self.useGpu.get() and iterNumber > 1:
            # Generate projections
            fnGalleryRoot = join(iterDir, "gallery")
            args = "-i %s -o %s.stk --sampling_rate %f --sym %s " \
                   "--compute_neighbors --angular_distance -1 " \
                   "--experimental_images %s --min_tilt_angle %f " \
                   "--max_tilt_angle %f -v 0 --perturb %f " % \
                   (prevVolFn, fnGalleryRoot, self.angularSampling.get(),
                    self.symmetryGroup, self.imgsFn, self.minTilt, self.maxTilt,
                    math.sin(self.angularSampling.get()) / 4)
            self.runJob("xmipp_angular_project_library ", args, numberOfMpi=1)

            if self.trueSymsNo != 0:
                alphaApply = (alpha * self.trueSymsNo) / 2
            else:
                alphaApply = alpha / 2
            from pwem.emlib.metadata import getSize
            N = int(getSize(fnGalleryRoot+'.doc')*alphaApply*2)

            count=0
            GpuListCuda=''
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuList = ' '.join([str(elem) for elem in self.getGpuList()])
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

            args = '-i %s -r %s.doc -o %s --keepBestN %f --dev %s ' % \
                   (self.imgsFn, fnGalleryRoot, anglesFn, N, GpuListCuda)
            self.runJob(CUDA_ALIGN_SIGNIFICANT, args, numberOfMpi=1)

            cleanPattern(fnGalleryRoot + "*")
        else:
            args = self.getSignificantArgs(self.imgsFn)
            args += ' --odir %s' % iterDir
            args += ' --alpha0 %f --alphaF %f' % (alpha, alpha)
            args += ' --dontCheckMirrors '

            if iterNumber == 1:
                if self.thereisRefVolume:
                    args += " --initvolumes " + \
                            self._getExtraPath('input_volumes.xmd')
                else:
                    args += " --numberOfVolumes 1"
            else:
                args += " --initvolumes %s" % prevVolFn

            self.runJob("xmipp_reconstruct_significant", args)
            moveFile(os.path.join(iterDir, 'angles_iter001_00.xmd'), anglesFn)
        t.toc('Significant took: ')

        reconsArgs = ' -i %s --fast' % anglesFn
        reconsArgs += ' -o %s' % volFn
        reconsArgs += ' --weight -v 0  --sym %s ' % self.symmetryGroup

        print("Number of images for reconstruction: ", metadata.getSize(
            anglesFn))
        t.tic()
        if self.useGpu.get():
            cudaReconsArgs = reconsArgs
            #AJ to make it work with and without queue system
            if self.numberOfMpi.get()>1:
                N_GPUs = len((self.gpuList.get()).split(','))
                cudaReconsArgs += ' -gpusPerNode %d' % N_GPUs
                cudaReconsArgs += ' -threadsPerGPU %d' % max(self.numberOfThreads.get(),4)
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
            cudaReconsArgs += ' --thr %s' %  self.numberOfThreads.get()
            if self.numberOfMpi.get()==1:
                cudaReconsArgs += ' --device %s' %(GpuListCuda)
            if self.numberOfMpi.get()>1:
                self.runJob('xmipp_cuda_reconstruct_fourier', cudaReconsArgs, numberOfMpi=len((self.gpuList.get()).split(','))+1)
            else:
                self.runJob('xmipp_cuda_reconstruct_fourier', cudaReconsArgs)
        else:
            self.runJob("xmipp_reconstruct_fourier_accel", reconsArgs)
        t.toc('Reconstruct fourier took: ')

        # Center the volume
        fnSym = self._getExtraPath('volumeSym_%03d.vol' % iterNumber)
        self.runJob("xmipp_transform_mirror", "-i %s -o %s --flipX" %
                    (volFn, fnSym), numberOfMpi=1)
        self.runJob("xmipp_transform_mirror", "-i %s --flipY" %
                    fnSym, numberOfMpi=1)
        self.runJob("xmipp_transform_mirror", "-i %s --flipZ" %
                    fnSym, numberOfMpi=1)
        self.runJob("xmipp_image_operate", "-i %s --plus %s" %
                    (fnSym, volFn), numberOfMpi=1)
        self.runJob("xmipp_volume_align", '--i1 %s --i2 %s --local --apply' %
                    (fnSym, volFn), numberOfMpi=1)
        cleanPath(fnSym)

        # To mask the volume
        xdim = self.inputSet.get().getDimensions()[0]
        maskArgs = "-i %s --mask circular %d -v 0" % (volFn, -xdim / 2)
        self.runJob('xmipp_transform_mask', maskArgs, numberOfMpi=1)
        # TODO mask the final volume in some smart way...

        # To filter the volume
        if self.useMaxRes:
            self.runJob('xmipp_transform_filter',
                        '-i %s --fourier low_pass %f --sampling %f' % \
                        (volFn, self.maxResolution.get(), self.TsCurrent),
                        numberOfMpi=1)

        if not self.keepIntermediate:
            cleanPath(prevVolFn, iterDir)

        if self.thereisRefVolume:
            cleanPath(self._getExtraPath('filteredVolume.vol'))

    def convertInputStep(self, classesFn):
        inputSet = self.inputSet.get()

        if isinstance(inputSet, SetOfClasses2D):
            writeSetOfClasses2D(inputSet, classesFn, writeParticles=False)
        else:
            writeSetOfParticles(inputSet, classesFn)

        # To re-sample input images
        fnDir = self._getExtraPath()
        fnNewParticles = join(fnDir, "input_classes.stk")
        TsOrig = self.inputSet.get().getSamplingRate()
        TsRefVol = -1
        if self.thereisRefVolume:
            TsRefVol = self.refVolume.get().getSamplingRate()
        if self.useMaxRes:
            self.TsCurrent = max([TsOrig, self.maxResolution.get(), TsRefVol])
            self.TsCurrent = self.TsCurrent / 3
            Xdim = self.inputSet.get().getDimensions()[0]
            self.newXdim = int(round(Xdim * TsOrig / self.TsCurrent))
            if self.newXdim < 40:
                self.newXdim = int(40)
                self.TsCurrent = float(TsOrig) * (
                        float(Xdim) / float(self.newXdim))
            if self.newXdim != Xdim:
                self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d"
                            % (self.imgsFn, fnNewParticles, self.newXdim),
                            numberOfMpi=self.numberOfMpi.get()
                                        * self.numberOfThreads.get())
            else:
                self.runJob("xmipp_image_convert", "-i %s -o %s "
                                                   "--save_metadata_stack %s"
                            % (self.imgsFn, fnNewParticles,
                               join(fnDir, "input_classes.xmd")), numberOfMpi=1)

        # To resample the refVolume if exists with the newXdim calculated
        # previously
        if self.thereisRefVolume:
            fnFilVol = self._getExtraPath('filteredVolume.vol')
            self.runJob("xmipp_image_convert", "-i %s -o %s -t vol" % (self.refVolume.get().getFileName(), fnFilVol),
                        numberOfMpi=1)
            # TsVol = self.refVolume.get().getSamplingRate()
            if self.useMaxRes:
                if self.newXdim != Xdim:
                    self.runJob('xmipp_image_resize', "-i %s --fourier %d" %
                                (fnFilVol, self.newXdim), numberOfMpi=1)
                    self.runJob('xmipp_transform_window', "-i %s --size %d" %
                                (fnFilVol, self.newXdim), numberOfMpi=1)
                    args = "-i %s --fourier low_pass %f --sampling %f " % (
                        fnFilVol, self.maxResolution.get(), self.TsCurrent)
                    self.runJob("xmipp_transform_filter", args, numberOfMpi=1)

            if not self.useMaxRes:
                inputVolume = self.refVolume.get()
            else:
                inputVolume = Volume(fnFilVol)
                inputVolume.setSamplingRate(self.TsCurrent)
                inputVolume.setObjId(self.refVolume.get().getObjId())
            fnVolumes = self._getExtraPath('input_volumes.xmd')
            row = metadata.Row()
            volumeToRow(inputVolume, row, alignType=ALIGN_NONE)
            md = emlib.MetaData()
            row.writeToMd(md, md.addObject())
            md.write(fnVolumes)

    def createOutputStep(self):
        lastIter = self.getLastIteration(1)
        Ts = self.inputSet.get().getSamplingRate()

        # To recover the original size of the volume if it was changed
        fnVol = self.getIterVolume(lastIter)
        Xdim = self.inputSet.get().getDimensions()[0]
        if self.useMaxRes and self.newXdim != Xdim:
            self.runJob('xmipp_image_resize', "-i %s --fourier %d" %
                        (fnVol, Xdim), numberOfMpi=1)
        fnMrc = fnVol.replace(".vol",".mrc")
        self.runJob("xmipp_image_convert","-i %s -o %s -t vol"%(fnVol,fnMrc),numberOfMpi=1)
        cleanPath(fnVol)
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(fnMrc,Ts),numberOfMpi=1)

        vol = Volume()
        vol.setObjComment('significant volume 1')
        vol.setLocation(fnMrc)
        vol.setSamplingRate(Ts)
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputSet, vol)

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if self.thereisRefVolume:
            if self.refVolume.hasValue():
                refVolume = self.refVolume.get()
                x1, y1, _ = refVolume.getDim()
                x2, y2, _ = self.inputSet.get().getDimensions()
                if x1 != x2 or y1 != y2:
                    errors.append('The input images and the reference volume '
                                  'have different sizes')
            else:
                errors.append("Please, enter a reference image")

        SL = emlib.SymList()
        SL.readSymmetryFile(self.symmetryGroup.get())
        if (100 - self.alpha0.get()) / 100.0 * (SL.getTrueSymsNo() + 1) > 1:
            errors.append("Increase the initial significance it is too low "
                          "for this symmetry")

        if self.useGpu and not isXmippCudaPresent():
            errors.append("You have asked to use GPU, but I cannot find Xmipp GPU programs in the path")
        return errors

    def _summary(self):
        summary = []
        summary.append("Input classes: %s" % self.getObjectTag('inputSet'))
        if self.thereisRefVolume:
            summary.append("Starting from: %s" % self.getObjectTag('refVolume'))
        else:
            summary.append("Starting from: 1 random volume")
        summary.append("Significance from %f%% to %f%% in %d iterations" %
                       (self.alpha0, self.alphaF, self.iter))
        if self.useImed:
            summary.append("IMED used")
        if self.strictDir:
            summary.append("Strict directions")
        return summary

    def _citations(self):
        return ['Sorzano2015']

    def _methods(self):
        retval = ""
        if self.inputSet.get() is not None:
            retval = "We used reconstruct significant to produce an " \
                     "initial volume "
            retval += "from the set of classes %s." % \
                      self.getObjectTag('inputSet')
            if self.thereisRefVolume:
                retval += " We used %s volume " % self.getObjectTag('refVolume')
                retval += "as a starting point of the reconstruction iterations."
            else:
                retval += " We started the iterations with 1 random volume."
            retval += " %d iterations were run going from a " % self.iter
            retval += "starting significance of %f%% to a final one of %f%%." % \
                      (self.alpha0, self.alphaF)
            if self.useImed:
                retval += " IMED weighting was used."
            if self.strictDir:
                retval += " The strict direction criterion was employed."

            if self.hasAttribute('outputVolume'):
                retval += " The reconstructed volume was %s." % \
                          self.getObjectTag('outputVolume')
        return [retval]

    # --------------------------- UTILS functions ------------------------------

    def getIterVolume(self, iterNumber):
        return self._getExtraPath('volume_iter%03d.vol' % iterNumber)

    def getIterTmpVolume(self, iterNumber):
        self._getTmpPath('iter%03d' % iterNumber, 'volume_iter001.vol')

    def getLastIteration(self, Nvolumes):
        lastIter = -1
        for n in range(1, self.iter.get() + 1):
            NvolumesIter = len(glob(self._getExtraPath('volume_iter%03d*.vol' % n)))
            if NvolumesIter == 0:
                continue
            elif NvolumesIter == Nvolumes:
                lastIter = n
            else:
                break
        return lastIter
