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
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtReconstructSignificant(ProtInitialVolume):
    """
    This algorithm addresses the initial volume problem in SPA
    by setting it in a Weighted Least Squares framework and
    calculating the weights through a statistical approach based on
    the cumulative density function of different image similarity measures.

    AI Generated

    ## Overview

    The Reconstruct Significant protocol generates an initial 3D volume from a set
    of 2D classes or averages using statistically significant angular assignments.

    Initial volume generation is one of the most delicate steps in single-particle
    cryo-EM. The goal is to obtain a plausible 3D map that can be used as a
    starting point for later refinement, without relying too strongly on an
    external reference. This protocol approaches the problem by assigning weights
    to possible image orientations according to statistical criteria and then
    reconstructing a volume from the most significant image-to-projection matches.

    The method starts with a relatively relaxed significance level and gradually
    moves toward a stricter one. Early iterations therefore explore a smoother and
    broader landscape of possible solutions, while later iterations focus on the
    more reliable angular assignments.

    The output is a reconstructed initial volume.

    ## Inputs and General Workflow

    The main input is a set of 2D classes or averages.

    The protocol converts the input images into Xmipp metadata format. If a
    reference volume is provided, it can be used to initialize the reconstruction.
    Otherwise, the protocol starts from a random volume.

    For each iteration, the protocol estimates significant angular assignments for
    the input images, reconstructs a volume using weighted Fourier reconstruction,
    centers the volume, masks it, and optionally filters it to a target resolution.

    The significance level changes progressively from the starting value to the
    final value over the selected number of iterations.

    ## Input Classes

    The **Input classes** parameter should point to a SetOfClasses2D or a set of
    averages.

    For class inputs, the class representatives are used as the images to
    reconstruct the initial volume. These class averages should be clean,
    representative, and preferably cover a broad range of particle views.

    The quality of the output depends strongly on the quality of the input 2D
    classes. If many classes are noisy, contaminated, duplicated, or inconsistent,
    the reconstructed volume may be distorted or unstable.

    Before using this protocol, it is usually advisable to remove clearly bad 2D
    classes.

    ## Symmetry Group

    The **Symmetry group** parameter defines the symmetry assumed during angular
    assignment and reconstruction.

    If the particle is asymmetric, use **c1**. If the structure has known symmetry,
    the corresponding Xmipp symmetry group can be specified.

    Correct symmetry can help stabilize reconstruction. However, imposing an
    incorrect symmetry can introduce artificial density and hide real asymmetric
    features.

    The protocol also checks that the initial significance is compatible with the
    selected symmetry. If the starting significance is too low for the symmetry
    group, the protocol asks the user to increase it.

    ## Reference Volume

    The option **Is there a reference volume(s)?** allows the user to provide an
    initial 3D reference volume.

    This reference can guide the first iteration. It may be useful when a very
    rough prior shape is known. For example, a cylindrical reference may help when
    working with fiber-like specimens. A symmetric reference may also be used as a
    starting point while reconstructing with lower or no imposed symmetry, allowing
    the result to break symmetry if supported by the data.

    The reference should be used carefully. If it is too detailed or biologically
    incorrect, it may bias the initial model. The safest use is as a coarse shape
    constraint rather than as a high-resolution template.

    If no reference volume is provided, the protocol starts from a random volume.

    ## Angular Sampling

    The **Angular sampling** parameter defines the angular step, in degrees, used
    to generate the projection gallery.

    A smaller value creates a denser angular search and can give more accurate
    orientation assignments, but increases computation time. A larger value is
    faster but may miss relevant orientations.

    For initial model generation, the default value is a reasonable starting point.
    Advanced users may adjust it depending on particle size, expected angular
    complexity, and available computational resources.

    ## Tilt Range

    The **Minimum tilt** and **Maximum tilt** parameters restrict the range of tilt
    angles considered during angular assignment.

    In this convention, 0 degrees corresponds to top views and 90 degrees to side
    views.

    Restricting the tilt range can be useful when the expected views are limited.
    For example, fiber-like specimens may be reconstructed mainly from side views,
    so the user may restrict the search around tilt angles close to 90 degrees.

    This option should be used only when the expected orientation distribution is
    known. Incorrect tilt restrictions can exclude valid views and bias the initial
    volume.

    ## Maximum Shift

    The **Maximum shift** parameter defines the allowed in-plane translation during
    the angular search, in pixels.

    If the value is set to **-1**, the shift search is unrestricted.

    A limited shift can make the alignment more stable when class averages are
    already centered. A freer shift search may be useful when class averages are
    not well centered, but it can also increase ambiguity.

    The value should reflect the expected centering accuracy of the input class
    averages.

    ## Starting and Final Significance

    The protocol uses two key significance parameters:

    - **Starting significance**;
    - **Final significance**.

    The starting significance defines how relaxed the first iterations are. A value
    such as 80% means that the protocol begins with a broad, smoother selection of
    significant matches.

    The final significance defines how strict the last iterations become. A value
    such as 99.5% means that only more statistically significant assignments have
    strong influence in the final reconstruction.

    This gradual change is important. Starting too strictly may leave too few
    images contributing to the reconstruction, producing noisy or unstable maps.
    Starting more relaxed allows the algorithm to explore a smoother solution
    space.

    ## Number of Iterations

    The **Number of iterations** parameter defines how many steps are used to move
    from the starting significance to the final significance.

    More iterations produce a more gradual transition. This may help the volume
    evolve more smoothly and avoid abrupt changes in angular assignment.

    Fewer iterations are faster but may make the progression from relaxed to strict
    criteria too abrupt.

    The default value is designed to provide a gradual refinement of the initial
    volume.

    ## Use IMED

    The **Use IMED** option enables IMED-based weighting.

    IMED is an image similarity measure that can be more discriminative than simple
    correlation when comparing very similar images. It can therefore help refine
    the weighting of angular assignments.

    This is an advanced option, but it is enabled by default because it can improve
    the statistical discrimination among candidate matches.

    ## Strict Direction

    The **Strict direction** option makes the angular-direction selection more
    selective.

    When this option is enabled, only the most significant experimental images are
    allowed to contribute to a given direction. This can produce a sharper and more
    selective reconstruction.

    However, it can also discard many experimental classes. In difficult datasets,
    only a small number of classes may contribute, making the reconstruction noisy.

    This option should be used carefully. It is useful when the user wants a
    stricter reconstruction, but it may be too aggressive for small or noisy input
    sets.

    ## Angular Neighborhood

    The **Angular neighborhood** parameter defines how neighboring directions
    contribute to the weighting of each image.

    The help text recommends using a value at least twice the angular sampling.
    This is because neighboring projections can provide useful contextual
    information when determining the statistical significance of an assignment.

    A larger neighborhood can make the weighting smoother. A smaller neighborhood
    makes the selection more local and possibly more sensitive to noise.

    ## Fisher Preselection

    The **Do not apply Fisher** option controls whether Fisher's confidence
    interval on the correlation coefficient is used for preselection.

    By default, images are preselected using this statistical criterion. This helps
    remove unreliable matches before reconstruction.

    If the option is enabled, this preselection is not applied. This may be useful
    for testing or specialized workflows, but most users should leave the default
    behavior unchanged.

    ## Maximum Resolution Option

    The **Use new maximum resolution?** option allows the user to simplify the
    calculation by keeping only low-frequency information.

    When enabled, the user specifies a **Target resolution** in angstroms. The
    input images and reference volume, if present, are resampled or filtered
    accordingly.

    This can reduce computation and make the initial volume search more robust by
    focusing on global structure rather than noisy high-resolution detail.

    For initial model generation, low- to medium-resolution information is usually
    more appropriate than high-frequency detail.

    ## Keep Intermediate Volumes

    The **Keep intermediate volumes** option controls whether intermediate volumes
    and angular assignments are preserved.

    Keeping intermediates is useful for debugging, method development, or detailed
    inspection of how the reconstruction evolves across iterations.

    Disabling this option saves disk space, which is usually preferable for routine
    use.

    ## GPU Execution

    The protocol can use GPU acceleration for significant angular assignment and
    Fourier reconstruction.

    GPU execution is enabled by default. If GPU execution is requested but the
    required Xmipp CUDA programs are not available, the protocol reports a
    validation error.

    GPU acceleration is especially useful because the protocol may generate
    projection galleries and perform many orientation-assignment and reconstruction
    steps.

    ## Volume Centering and Masking

    After each reconstruction, the protocol performs centering operations and
    applies a circular mask.

    The centering step compares the volume with its mirrored version and locally
    aligns it. This helps keep the reconstructed density centered during
    iterations.

    The circular mask removes density outside the expected volume support and
    reduces the influence of peripheral noise.

    These operations are part of the internal stabilization of the reconstruction.

    ## Output Volume

    The main output is **outputVolume**.

    This is the final reconstructed volume from the last completed iteration. The
    volume is converted to MRC format and registered in Scipion with the sampling
    rate of the input image set.

    The output volume should be interpreted as an initial model suitable for later
    3D refinement, not as a final high-resolution reconstruction.

    ## Interpreting the Result

    The reconstructed volume represents the structure supported by statistically
    significant matches between the input 2D images and projections of the evolving
    3D map.

    A good result should show a plausible global shape consistent with the 2D
    classes. Fine details should not be overinterpreted at this stage.

    Poor results may occur if the input classes are noisy, if there are too few
    representative views, if the symmetry is wrong, if the significance criteria
    are too strict, or if a reference volume biases the search incorrectly.

    ## Practical Recommendations

    Use clean and representative 2D class averages as input. Remove obvious junk
    classes before running the protocol.

    Use **c1** unless the symmetry is known and biologically justified.

    Start with the default significance schedule. If the reconstruction is too
    noisy, the final significance may be too strict or too few classes may be
    contributing.

    Use a reference volume only as a rough guide. Avoid highly detailed references
    that could bias the initial model.

    Use tilt restrictions only when the expected orientation range is known, such
    as side-view-dominated fiber datasets.

    Enable the maximum-resolution option when you want to focus the search on
    low-resolution shape and reduce the influence of noise.

    Inspect the final volume as an initial model and validate it through subsequent
    refinement and comparison with the input 2D classes.

    ## Final Perspective

    Reconstruct Significant is an initial-volume generation protocol based on
    statistically weighted angular assignments.

    For biological users, its value is that it can produce a plausible starting
    volume from 2D classes or averages while controlling which image-to-projection
    matches are allowed to influence the reconstruction.

    The protocol is most useful when the input classes are clean, the symmetry and
    tilt-range assumptions are appropriate, and the output is treated as a
    starting point for further refinement rather than as a final map.
    """
    _label = 'reconstruct significant'
    _devStatus = PROD

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
        self._insertFunctionStep(self.convertInputStep, self.imgsFn, )
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

            gpuList = list(map(str, self._stepsExecutor.getGpuList()))
            gpuListArg=" ".join(gpuList)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpuList)
            self.info("GPUs used in CUDA_VISIBLE_DEVICES: %s" % gpuListArg)

            cudaReconsArgs += ' --thr %s' % self.numberOfThreads.get()
            if self.numberOfMpi.get() == 1:
                cudaReconsArgs += ' --device ' + gpuListArg
            if self.numberOfMpi.get() > 1:
                self.runJob('xmipp_cuda_reconstruct_fourier', cudaReconsArgs, numberOfMpi=len(gpuList)+1)
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
