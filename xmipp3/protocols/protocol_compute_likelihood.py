# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
# *              James Krieger        (jamesmkrieger@gmail.com)
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

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
import os

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, SetOfParticles, SetOfClasses3D
from pwem import emlib
import pwem.emlib.metadata as md

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, EnumParam, FloatParam,
                                        BooleanParam, IntParam, 
                                        USE_GPU, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL


from xmipp3.convert import setXmippAttributes, writeSetOfParticles
import xmippLib
from pyworkflow import BETA, UPDATED, NEW, PROD



class XmippProtComputeLikelihood(ProtAnalysis3D):
    """This protocol computes the log likelihood or correlation of a set of
    particles with assigned angles when compared to a set of maps or atomic
    models

    AI Generated:

    Overview

    The Log Likelihood protocol evaluates how well experimental cryo-EM particle
     images agree with one or more reference structures. These references can
     be 3D volumes obtained from previous reconstructions or sets of volumes
     representing different structural states. For each particle, the protocol
     computes a statistical measure—typically a log likelihood score—that
     quantifies how consistent the particle image is with the projections of
     each reference.

    In practice, this protocol answers a biologically meaningful question:

    Given the current particle orientations, which structural model best
    explains each particle image?

    This type of analysis is useful in several situations. It can help compare
    particles against multiple candidate structures, quantify the quality of
    particle assignments, or evaluate how well experimental data support a
    given model. The protocol can also automatically group particles according
    to the reference that best explains them, providing a simple form of
    likelihood-based classification.

    Because the protocol relies on already assigned particle orientations, it
    is typically applied after alignment or refinement steps in the cryo-EM
    workflow.

    Inputs and General Workflow

    The protocol requires two main inputs:

    Input particles

    These must be particle images with assigned projection angles. In other
    words, each particle should already have orientation parameters describing
    how it relates to a 3D structure. These orientations usually come from
    previous reconstruction or refinement steps.

    Reference volumes

    The user can provide either:
    - a single volume, or
    - a set of volumes representing different structural hypotheses or
    conformational states.

    For each particle, the protocol generates projections of the reference
    volume(s) using the particle’s assigned orientation and compares them with
    the experimental image.

    The comparison produces a log likelihood score, reflecting how probable the
    experimental image is given the reference projection and the estimated
    noise. When multiple references are provided, the protocol compares each
    particle against all references and determines which one best explains the
    data.

    Particle and Noise Regions

    A key concept in this protocol is the separation between particle signal
    and background noise. The algorithm estimates the noise statistics from a
    region surrounding the particle and uses this information when computing
    the likelihood score.

    Two parameters control this separation.

    Particle radius. This defines the circular region that contains the particle signal.
    Ideally, this radius should include the full particle but avoid large
    solvent regions. Choosing a value that is too large may dilute the signal,
    while a value that is too small may exclude important structural features.

    If this parameter is left at the default value, the protocol assumes that
    the particle occupies roughly half the image width.

    Noise radius. This defines the outer radius used to estimate background noise. The
    region between the particle radius and the noise radius forms a ring where
    the algorithm measures noise statistics.

    In practice:

    The particle radius should cover the particle.

    The noise radius should extend slightly beyond the particle.

    If the noise radius is not specified, the protocol automatically uses the
    outer image region.

    Correct estimation of noise is important because it strongly affects the
    likelihood calculation.

    Reference Projections and Residual Images

    For each particle and each reference volume, the protocol generates a
    projection of the reference using the particle’s assigned orientation.
    This projection is then compared with the experimental image.

    The difference between the projection and the experimental particle image
    is called the residual image. Residuals contain information about:
    - noise in the experimental image
    - model inaccuracies
    - structural differences between particle and reference

    These residuals are used internally to estimate the likelihood score and
    can optionally be stored for further analysis.

    From a biological perspective, large residuals often indicate that the
    particle is poorly explained by the reference model.

    Gray Level Optimization

    Experimental particle images and simulated projections may differ in
    overall intensity scale. The protocol therefore includes an option to
    optimize the gray scale factor between the projection and the experimental
    image.

    When enabled, the algorithm adjusts the intensity scale to maximize the
    agreement between the two images. This step improves robustness when image
    normalization or detector scaling differs across datasets.

    A parameter called maximum gray change limits how much the scale can vary.
    Restricting this range prevents unrealistic intensity adjustments that
    could artificially inflate the likelihood score.

    In most practical cases, enabling gray optimization improves stability and
    is recommended.

    Normalization Options

    The protocol optionally performs intensity normalization of both particles
    and reference projections. Normalization ensures that the likelihood
    calculation is not biased by global intensity differences.

    Three normalization strategies are available:

    Old Xmipp normalization

    The entire image is normalized so that the mean intensity becomes zero and
    the standard deviation becomes one.

    New Xmipp normalization

    Normalization is performed using only the background region of the image.
    This approach better preserves the particle signal and is generally
    preferred for cryo-EM analysis.

    Ramp normalization

    This method subtracts background gradients before applying normalization.
    It is particularly useful when micrographs contain slow intensity
    variations or illumination gradients.

    For most cryo-EM workflows, the newer normalization approaches provide
    more reliable statistical behavior.

    CTF Considerations

    The protocol can optionally ignore the Contrast Transfer Function (CTF)
    during the comparison between particles and projections.

    In general, CTF effects should be considered during likelihood calculations
    because they influence the appearance of particle images. However, this
    option becomes useful when images have already been CTF-corrected using
    Wiener filtering or similar procedures.

    If the dataset has already undergone strong CTF correction, disabling CTF
    application can avoid redundant processing.

    Outputs and Their Interpretation

    After execution, the protocol produces several outputs.

    Particle set with likelihood values

    Each particle receives a log likelihood score describing how well it
    matches each reference. These values can be used to evaluate particle
    quality or to study the consistency between particles and structural models.

    Residual images (optional)

    Residual images represent the difference between experimental particles
    and reference projections. Inspecting residuals can reveal systematic
    mismatches, noise patterns, or structural variability.

    Likelihood matrix

    Internally, the protocol constructs a matrix containing the likelihood of
    every particle with respect to every reference. This matrix provides a
    quantitative view of how strongly each particle supports each structural
    hypothesis.

    3D classes based on likelihood

    When multiple references are provided, the protocol assigns each particle
    to the reference with the highest likelihood score. The resulting grouping
    is stored as a set of 3D classes, where each class corresponds to one
    reference volume.

    Biologically, this allows users to see which particles are most compatible
    with each structural state.

    Practical Recommendations

    In most workflows, this protocol is applied after particle orientations
     have been determined, typically following a refinement or reconstruction
     step.

    A common use case is to compare particles against several candidate models
    representing different conformations or processing strategies. In this
    situation, the likelihood scores provide a quantitative way to determine
    which model best explains the experimental data.

    When selecting the particle radius, it is usually better to slightly
    overestimate the particle size rather than risk excluding relevant
    structural features. However, the noise ring should still contain a
    reasonable background region for estimating noise variance.

    If the dataset has inconsistent intensity scaling, enabling gray
    optimization usually improves the reliability of the results.

    When analyzing heterogeneous datasets, the likelihood-based classification
    produced by the protocol can serve as an initial indicator of structural
    variability. However, it should not replace dedicated classification
    methods, which are typically more sensitive to subtle conformational
    differences.

    Final Perspective

    Likelihood evaluation provides a statistically grounded way to connect
    experimental particle images with structural models. Rather than relying
    solely on visual inspection or correlation scores, the log likelihood
    measures how probable each particle image is under a given structural
    hypothesis.

    For cryo-EM users, this protocol offers a powerful tool for model
    validation, particle quality assessment, and reference comparison,
    helping ensure that biological interpretations are supported by the
    underlying experimental data.
    """

    _label = 'log likelihood'
    _lastUpdateVersion = VERSION_1_1
    _possibleOutputs = {"reprojections": SetOfParticles}
    _devStatus = PROD
    stepsExecutionMode = STEPS_PARALLEL

    # Normalization enum constants
    NORM_OLD = 0
    NORM_NEW = 1
    NORM_RAMP =2

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParam('binThreads', IntParam,
                      label='threads',
                      default=2,
                      help='Number of threads used by Xmipp each time it is called in the protocol execution. For '
                           'example, if 3 Scipion threads and 3 Xmipp threads are set, the particles will be '
                           'processed in groups of 2 at the same time with a call of Xmipp with 3 threads each, so '
                           '6 threads will be used at the same time. Beware the memory of your machine has '
                           'memory enough to load together the number of particles specified by Scipion threads.')
        
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label="Input images", important=True,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignmentProj')
        form.addParam('inputRefs', PointerParam, label="References", important=True,
                      pointerClass='Volume,SetOfVolumes',
                      help='Volume or set of volumes to which the set of particles will be compared')
        form.addParam('particleRadius', IntParam, label="Particle radius (px): ", default=-1,
                      help='This radius should include the particle but be small enough to leave room to create a ring for estimating noise\n'
                            'If left at -1, this will take half the image width. In this case, the whole circle will be used to estimate noise')
        form.addParam('noiseRadius', IntParam, label="Noise radius (px): ", default=-1,
                      help='This radius should be larger than the particle radius to create a ring for estimating noise\n'
                            'If left at -1, this will take half the image width.')

        form.addParam('newProg', BooleanParam, label="Use new program: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Whether to use new program xmipp_continuous_create_residuals. This removes the low-pass filter and '
                      'applies transformations to the projection, not the original image.')

        form.addParam('optimizeGray', BooleanParam, label="Optimize gray: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Optimize the gray value between the map reprojection and the experimental image')
        form.addParam('maxGrayChange', FloatParam, label="Max. gray change: ", default=0.99, expertLevel=LEVEL_ADVANCED,
                      condition='optimizeGray',
                      help='The actual gray value can be at most as small as 1-change or as large as 1+change')

        form.addParam('doNorm', BooleanParam, default=False,
                      label='Normalize', expertLevel=LEVEL_ADVANCED,
                      help='Whether to subtract background gray values and normalize '
                           'so that in the background there is 0 mean and '
                           'standard deviation 1. This is applied to particles and volumes')
        form.addParam('normType', EnumParam, condition='doNorm',
                      label='Particle normalization type', expertLevel=LEVEL_ADVANCED,
                      choices=['OldXmipp','NewXmipp','Ramp'],
                      default=self.NORM_RAMP, display=EnumParam.DISPLAY_COMBO,
                      help='OldXmipp: mean(Image)=0, stddev(Image)=1\n'
                           'NewXmipp: mean(background)=0, stddev(background)=1\n'
                           'Ramp: subtract background + NewXmipp\n'
                           'Only New and Old Xmipp are supported for volumes.'
                           'If Ramp is selected then New is used for volumes')

        form.addParam('ignoreCTF', BooleanParam, label="Do not apply CTF: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='This should be used when images are treated with a Weiner filter instead')

        form.addParam('printTerms', BooleanParam, label="Print terms of LL: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to print terms 1 and 2, LL and noise variance')

        form.addParallelSection(threads=2, mpi=2)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ Convert input images then run continuous_assign2 then create output """
        
        self.imagesXmd = self._getExtraPath("images.xmd")
        self.imagesStk = self._getExtraPath("images.stk")

        convId = self._insertFunctionStep(self.convertStep, prerequisites=[], needsGPU=False)
        stepIds = [convId]
        if self.doNorm:
            normPsId = self._insertFunctionStep(self.normalizeParticlesStep, prerequisites=convId,
                                                needsGPU=False)
            stepIds.append(normPsId)

        inputRefs = self.inputRefs.get()
        i=1
        if isinstance(inputRefs, Volume):
            prodId = self._insertFunctionStep(self.produceResidualsStep, inputRefs.getFileName(), i,
                                              prerequisites=stepIds, needsGPU=False)
            i += 1
            stepIds.append(prodId)
        else:
            for volume in inputRefs:
                prodId = self._insertFunctionStep(self.produceResidualsStep, volume.getFileName(), i,
                                                  prerequisites=stepIds, needsGPU=False)
                i += 1
                stepIds.append(prodId)

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=stepIds,
                                 needsGPU=False)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        imgSet = self.inputParticles.get()
        writeSetOfParticles(imgSet, self.imagesXmd)

    def normalizeParticlesStep(self):
        argsNorm = self._argsNormalize(particles=True)
        self.runJob("xmipp_transform_normalize", argsNorm)

    def getMasks(self):
        if not (hasattr(self, 'mask') and hasattr(self, 'noiseMask')):
            Xdim = self._getSize()
            Y, X = np.ogrid[:Xdim, :Xdim]
            dist_from_center = np.sqrt((X - Xdim/2) ** 2 + (Y - Xdim/2) ** 2)

            particleRadius = self.particleRadius.get()
            if particleRadius<0:
                particleRadius=Xdim/2
            self.mask = dist_from_center <= particleRadius

            noiseRadius = self.noiseRadius.get()
            if noiseRadius == -1:
                noiseRadius = Xdim/2
            if noiseRadius > particleRadius:
                self.noiseMask = (dist_from_center > particleRadius) & (dist_from_center <= noiseRadius)
            else:
                self.noiseMask = self.mask

        return self.mask, self.noiseMask

    def produceResidualsStep(self, fnVol, i):

        if self.doNorm:
            fnVolOut = self._getExtraPath('%03d_' % (i) + os.path.split(fnVol)[1])
            argsNorm = "-i %s -o %s" % (fnVol, fnVolOut) + self._argsNormalize()
            self.runJob("xmipp_transform_normalize", argsNorm)
            fnVol = fnVolOut

        if self.newProg:
            anglesOutFn = self._getExtraPath("anglesCont%03d.xmd"%i)
            prog = "xmipp_continuous_create_residuals"
        else:
            anglesOutFn = self._getExtraPath("anglesCont%03d.stk"%i)
            prog = "xmipp_angular_continuous_assign2"

        fnResiduals = self._getExtraPath("residuals%03d.stk"%i)
        fnProjections = self._getExtraPath("projections%03d.stk"%i)

        Ts = self.inputParticles.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --sampling %f --oresiduals %s --oprojections %s" % (self.imagesXmd, anglesOutFn,
                                                                                         fnVol, Ts,
                                                                                         fnResiduals, fnProjections)

        if self.optimizeGray:
            args+=" --optimizeGray --max_gray_scale %f"%self.maxGrayChange

        self.runJob(prog, args, numberOfMpi=self.numberOfMpi.get())

        mdResults = md.MetaData(self._getExtraPath("anglesCont%03d.xmd"%i))
        mdOut = md.MetaData()

        if self.printTerms.get():
                print('{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\n'.format('-sos', 'term1', 'term2',
                                                                                 'LL', 'var', '1/(2*var)', 'std'))
        for objId in mdResults:
            itemId = mdResults.getValue(emlib.MDL_ITEM_ID, objId)

            if self.optimizeGray:
                fnResidual = mdResults.getValue(emlib.MDL_IMAGE_RESIDUAL, objId)
                I = xmippLib.Image(fnResidual)

                elements_within_circle = I.getData()[self.getMasks()[0]]
                sum_of_squares = np.sum(elements_within_circle**2)
                Npix = elements_within_circle.size

                elements_between_circles = I.getData()[self.getMasks()[1]]
                var = np.var(elements_between_circles)

                term1 = -sum_of_squares/(2*var)
                term2 = Npix/2 * np.log(2*np.pi*var)
                LL = term1 - term2

                if self.printTerms.get():
                    print('{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\n'.format(-sum_of_squares, term1, term2,
                                                                                                    LL, var, 1/(2*var), var**0.5))

            else:
                LL = mdResults.getValue(emlib.MDL_COST, objId)

            newRow = md.Row()
            newRow.setValue(emlib.MDL_ITEM_ID, itemId)
            newRow.setValue(emlib.MDL_LL, float(LL))
            newRow.setValue(emlib.MDL_IMAGE_REF, fnVol)

            if self.optimizeGray:
                newRow.setValue(emlib.MDL_IMAGE_RESIDUAL, fnResidual)

            newRow.addToMd(mdOut)
        mdOut.write(self._getLLfilename(i))

    def appendRows(self, outputSet, fnXmd):
        self.iterMd = md.iterRows(fnXmd, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(self.inputParticles.get(), updateItemCallback=self._processRow)

    def createOutputStep(self):
        inputPartSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputPartSet)

        refsDict = {}
        i=1
        if isinstance(self.inputRefs.get(), Volume):
            self.appendRows(outputSet, self._getLLfilename(i))
            refsDict[i] = self.inputRefs.get()
            i += 1
        else:
            for volume in self.inputRefs.get():
                self.appendRows(outputSet, self._getLLfilename(i))
                refsDict[i] = volume.clone()
                i += 1

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        matrix = np.array([particle._xmipp_logLikelihood.get() for particle in outputSet])
        matrix = matrix.reshape((i-1,-1))
        np.save(self._getExtraPath('matrix.npy'), matrix)

        classIds = np.argmax(matrix, axis=0)+1

        clsSet = SetOfClasses3D.create(self._getExtraPath())
        clsSet.setImages(inputPartSet)

        clsDict = {}  # Dictionary to store the (classId, classSet) pairs

        for ref, rep in refsDict.items():
            # add empty classes
            classItem = clsSet.ITEM_TYPE.create(self._getExtraPath(), suffix=ref+1)
            classItem.setRepresentative(rep)
            clsDict[ref] = classItem
            clsSet.append(classItem)

        cls_prev = 1
        for img, ref in izip(inputPartSet, classIds):
            if ref != cls_prev:
                cls_prev = ref

            classItem = clsDict[ref]
            classItem.append(img)

        for classItem in clsDict.values():
            clsSet.update(classItem)

        clsSet.write()

        self._defineOutputs(outputClasses=clsSet)
        self._defineSourceRelation(self.inputParticles, clsSet)
        self._defineSourceRelation(self.inputRefs, clsSet)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise ValueError("Missing row %s at %s" % (id, mdFile))

    def _processRow(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                particle.setObjId(None)
                setXmippAttributes(particle, self.lastRow,
                                   emlib.MDL_LL, emlib.MDL_IMAGE_REF,
                                   emlib.MDL_IMAGE_RESIDUAL)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _argsNormalize(self, particles=False):
        args = ""
        if particles:
            args += "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.imagesXmd, self.imagesStk, self.imagesXmd)

        normType = self.normType.get()
        bgRadius = self.particleRadius.get()
        radii = self._getSize()
        if bgRadius <= 0:
            bgRadius = int(radii)

        if normType == self.NORM_OLD:
            args += " --method OldXmipp"
        elif normType == self.NORM_NEW or not particles:
            args += " --method NewXmipp --background circle %d" % bgRadius
        else:
            args += " --method Ramp --background circle %d" % bgRadius
        return args

    def _getSize(self):
        return self.inputParticles.get().getDimensions()[0]

    def _getLLfilename(self, i):
        return self._getExtraPath("logLikelihood%03d.xmd" % i)
