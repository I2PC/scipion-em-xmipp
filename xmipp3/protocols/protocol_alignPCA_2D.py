# ******************************************************************************
# *
# * Authors: Erney Ramirez Aportela (eramirez@cnb.csic.es)
# *          Daniel Marchan Torres (da.marchan@cnb.csic.es)
# *          Yunior C. Fonseca Reyna (cfonseca@cnb.csic.es)
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
# ******************************************************************************
import enum
import json
import sys
import emtable
import os

import pyworkflow.utils as pwutils
import time
import numpy as np

from pwem.protocols import ProtClassify2D
from pyworkflow import VERSION_3_0
from pyworkflow.object import Set, RELATION_SOURCE
from pyworkflow.protocol.params import IntParam, StringParam, PointerParam, EnumParam, BooleanParam, FloatParam
from pyworkflow.protocol import ProtStreamingBase, STEPS_PARALLEL, GPU_LIST, LEVEL_ADVANCED, MODE_RESUME
from pyworkflow.constants import BETA

from pwem.objects import SetOfClasses2D, SetOfAverages, Transform
from pwem.constants import ALIGN_NONE, ALIGN_2D, ALIGN_PROJ, ALIGN_3D
from xmipp3.base import XmippProtocol
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

from xmipp3.convert import (readSetOfParticles, writeSetOfParticles,
                            writeSetOfClasses2D, xmippToLocation, matrixFromGeometry)

OUTPUT_CLASSES = "outputClasses"
OUTPUT_AVERAGES = "outputAverages"


class XMIPPCOLUMNS(enum.Enum):
    # PARTICLES CONSTANTS
    ctfVoltage = "ctfVoltage"  # 1
    ctfDefocusU = "ctfDefocusU"  # 2
    ctfDefocusV = "ctfDefocusV"  # 3
    ctfDefocusAngle = "ctfDefocusAngle"  # 4
    ctfSphericalAberration = "ctfSphericalAberration"  # 5
    ctfQ0 = "ctfQ0"  # 6
    ctfCritMaxFreq = "ctfCritMaxFreq"  # 7
    ctfCritFitting = "ctfCritFitting"  # 8
    enabled = "enabled"  # 9
    image = "image"  # 10
    itemId = "itemId"  # 11
    micrograph = "micrograph"  # 12
    micrographId = "micrographId"  # 13
    scoreByVariance = "scoreByVariance"  # 14
    scoreByGiniCoeff = "scoreByGiniCoeff"  # 15
    xcoor = "xcoor"  # 16
    ycoor = "ycoor"  # 17
    ref = "ref"  # 18
    anglePsi = "anglePsi"  # 19
    angleRot = "angleRot"  # 20
    angleTilt = "angleTilt"  # 21
    shiftX = "shiftX"  # 22
    shiftY = "shiftY"  # 23
    shiftZ = "shiftZ"  # 24
    flip = "flip"

    # CLASSES CONSTANTS
    classCount = "classCount"  # 3


ALIGNMENT_DICT = {"shiftX": XMIPPCOLUMNS.shiftX.value,
                  "shiftY": XMIPPCOLUMNS.shiftY.value,
                  "shiftZ": XMIPPCOLUMNS.shiftZ.value,
                  "flip": XMIPPCOLUMNS.flip.value,
                  "anglePsi": XMIPPCOLUMNS.anglePsi.value,
                  "angleRot": XMIPPCOLUMNS.angleRot.value,
                  "angleTilt": XMIPPCOLUMNS.angleTilt.value
                  }


def updateEnviron(gpuNum):
    """ Create the needed environment for pytorch programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

CONTRAST_AVERAGES_FILE = 'classes_classes.star'
AVERAGES_IMAGES_FILE = 'classes_images.star'



class _LogicalParticleBatchView:
    """In-memory view of one logically reconstructed particle batch."""

    def __init__(self, particles, alignment):
        self._particles = particles
        self._alignment = alignment

    def __iter__(self):
        return iter(self._particles)

    def __len__(self):
        return len(self._particles)

    def getAlignment(self):
        return self._alignment


class XmippProtClassifyPcaStreaming(XmippStreamingBase, ProtStreamingBase, ProtClassify2D, XmippProtocol):
    """ Performs a 2D classification of particles using PCA. This method is optimized to run in streaming,
        enabling efficient processing of large datasets.

        AI Generated:

        What this protocol is for

        alignPCA-2D performs 2D alignment and classification of single-particle
        images using a PCA-based approach, and it is designed to work
        especially well in streaming. The idea is that as particles are being
        produced upstream (for example, from particle picking/extraction
        running in streaming), this protocol can periodically take the newest
        particles, classify them into 2D classes, and continuously update the
        output classes. For a biological user, the practical benefit is that
        you can start getting meaningful 2D class averages early, monitor data
        quality in near real time, and progressively refine the class set as
        more data arrives, instead of waiting for the full dataset to finish.

        This protocol can be used in two distinct ways. In the create classes
        mode, it builds a new 2D class set from scratch. In the update classes
        mode, it does not “invent” new references, but instead aligns and
        assigns incoming particles to a set of pre-existing 2D classes (either
        a SetOfClasses2D or a SetOfAverages). That second mode is particularly
        useful when you want continuity: you already trust a set of references
        and you want to keep updating them or at least keep assigning particles
        consistently as the stream grows.

        Inputs and what they should contain

        You provide a single SetOfParticles as input images. Since the protocol
        can optionally perform CTF correction internally, it expects your
        particles to have the usual CTF metadata available (defoci, voltage,
        Cs, etc.) if you plan to enable CTF correction. In practice, this
        protocol is most comfortable with particle boxes that are not too
        large; as a rule of thumb, particle sizes around 128 px or smaller
        are recommended for stable GPU memory use and speed. If you run it on
        much larger boxes (especially above 256 px), you risk saturating GPU
        memory, and performance may degrade substantially. Biologically, this
        is rarely a limitation because 2D classification is typically done on
        binned or otherwise downscaled particles; you can always classify on
        smaller boxes first to obtain clean 2D averages and then return to
        larger boxes later for high-resolution refinement steps.

        GPU selection and compute behaviour in streaming

        Because the core PCA-based alignment/classification is GPU-accelerated,
        you will choose a GPU ID. If you only have one GPU or you are unsure,
        leaving the default is usually fine. In a facility context, the
        important point is to avoid collisions with other GPU-heavy protocols:
        pick a GPU that is actually free.

        In streaming, the protocol processes particles in batches. It
        periodically checks the input set for newly created particles (using
        their creation time), groups them into a growing “new particles”
        batch, and launches classification whenever the batch is large enough
        or when the input stream is closed and it needs to flush the final
        incomplete batch. For the user, this means you will see your 2D classes
        update stepwise rather than continuously particle-by-particle. This is
        generally desirable: it avoids constant re-training and keeps
        throughput high.

        Choosing “create classes” versus “update classes”

        If you choose create_classes, you specify the number of classes you
        want. This controls how finely the dataset will be partitioned. From a
        biological standpoint, the right number depends on heterogeneity and
        dataset size. A smaller number of classes (for example 50) is a
        reasonable default for monitoring and for early cleaning; larger values
        can separate subtle views or different particle populations but may
        also produce more low-occupancy classes, especially early in streaming
        when the dataset is still small.

        If you choose update_classes, you must provide initial classes. This is
        ideal when you already have a trusted set of 2D references—perhaps from
        a previous run, a curated subset, or a high-quality earlier
        classification—and you want new data to be aligned and assigned in a
        way that remains consistent with those references. Biologically, this
        is very useful when you want stable class labels across time, for
        instance during long microscope sessions or when comparing multiple
        acquisition sessions against a common reference set.

        CTF correction: why it matters and when to enable it

        The option Correct CTF? controls whether the protocol internally
        performs a Wiener-like 2D CTF correction before classification. For
        most biological users, enabling CTF correction is beneficial because
        it makes particles more comparable in Fourier space across defocus
        values and typically yields cleaner, more interpretable class averages.
        This is especially noticeable when defocus ranges are broad or when
        you care about higher-resolution features in the 2D averages.

        The practical trade-off is compute: CTF correction can be expensive.
        The protocol therefore allows you to increase MPIs specifically for the
        CTF correction step. If you have many CPU cores available, increasing
        this MPI value can speed up the preprocessing significantly. In
        facility environments, this is often worth doing because it can keep
        the GPU from idling while waiting for preprocessed particles.

        If you disable CTF correction, the protocol will classify using the
        raw particle images (converted as-is). This can still be useful for
        very fast, rough monitoring or when CTF metadata is unreliable, but
        in many cryo-EM datasets you will obtain more stable classes with
        correction enabled.

        Gaussian mask: focusing alignment on the particle signal

        The option Use Gaussian Mask? applies a soft, Gaussian-shaped mask
        to the particle images. Biologically, masking is often beneficial
        in 2D classification because it reduces the impact of noisy background
        and edge artifacts introduced by boxing. This tends to stabilize
        alignments and can prevent classes from being driven by irrelevant
        high-contrast features near the box border.

        The key parameter is sigma, which controls the width of the Gaussian.
        If you leave sigma at −1, the protocol automatically sets sigma to
        approximately one third of the image dimension. That default is usually
        sensible for general particles. As a biological user, you might
        consider adjusting sigma when you have unusually large boxes with lots
        of solvent, or when the particle occupies only a small central region;
        in those cases, a tighter mask can sometimes yield cleaner averages.
        Conversely, if you fear truncation of peripheral features (for example,
        extended domains or detergent belts), a broader mask can be safer.

        PCA training: what the parameters mean in practice

        This protocol uses PCA to build a compact representation of the
        dataset that supports fast alignment and classification. For the user,
        the PCA-related parameters mainly affect a balance between speed,
        stability, and detail.

        The max resolution parameter sets the maximum resolution content
        considered during alignment. Conceptually, it prevents the algorithm
        from chasing very high-frequency details that are not stable at the
        current SNR or that are not needed for robust 2D grouping. Biologically,
        this is helpful because 2D classification is typically aimed at
        separating views and major structural features rather than extracting
        the final high-resolution signal. A value around 8 Å is a common choice
        for stable alignment in many datasets, but the best setting depends on
        pixel size and data quality. Importantly, the protocol ensures that
        this value is not set unrealistically high relative to the sampling:
        it will enforce a sensible lower bound tied to the pixel size.

        The % variance parameter determines how much variance the PCA model
        should capture. Higher values generally mean the PCA representation
        keeps more subtle differences, which can improve accuracy in
        distinguishing similar views or small conformational changes, but it
        increases computation and can make the model more sensitive to noise.
        For most biological workflows, you can keep this around the default
        unless you have a specific reason: raising it may help when you want
        more detailed separation; lowering it can speed up early monitoring.

        The particles for training parameter controls how many particles are
        used to build the PCA model. In streaming, this is particularly
        important: too small a training set may produce a fragile PCA basis
        early on, while too large a value increases the initial cost. In
        practical biological usage, it is often fine to start with a moderate
        training size so that the protocol begins producing classes early, and
        then let later rounds refine as more particles arrive.

        Classification in streaming: batch size and what you will see

        The parameter particles for initial classification in streaming
        defines the batch size threshold that triggers a classification round.
        Biologically, this controls the “update rhythm” of your classes. A
        smaller batch size yields more frequent updates and earlier feedback,
        but each update is based on fewer particles, so early class averages
        may look noisier or may fluctuate. A larger batch size gives more
        stable updates but delays feedback.

        A good way to think about it is monitoring versus stability. During
        acquisition, many users prefer earlier feedback, so a moderate batch
        size is often best. Once acquisition stabilizes or when running
        offline, you might increase the batch size to reduce overhead and
        produce more stable class updates.

        Outputs: what you get and how to interpret it biologically

        The protocol produces a streaming SetOfClasses2D as the main output.
        As new particles are processed, the class set gets updated: particles
        receive class assignments and in-plane alignment parameters, and each
        class gets an updated representative image (the class average). This
        output is what you will inspect to judge particle quality, preferred
        orientations, contamination, aggregation, and general dataset
        heterogeneity.

        A secondary output type may be a SetOfAverages, depending on the
        workflow and how downstream consumers use the references, but the
        main biological deliverable is the evolving 2D class set.

        When interpreting the results during streaming, it is normal that
        early rounds show unstable or low-quality averages. The most useful
        early signal is usually whether recognizable projections appear at
        all and whether junk classes dominate. As more particles accumulate,
        good classes should become sharper, class occupancy should become
        more meaningful, and rare views may begin to appear.

        Practical usage patterns (biological perspective)

        A very common workflow is to run this protocol in streaming during
        acquisition with CTF correction enabled and the Gaussian mask enabled,
        using a moderate number of classes. This provides rapid, biologically
        interpretable feedback: you can see whether the sample is good, whether
        there are multiple particle populations, whether the dataset is
        dominated by ice contamination or carbon edges, and whether the
        microscope session is producing useful projections.

        Another common pattern is to first run create_classes to obtain a
        clean reference set, curate the best classes (removing obvious junk),
        and then restart in update_classes mode using those curated classes as
        initial references. This is especially useful for long sessions or
        multiple sessions, because it stabilizes class identity across time
        and makes comparisons much easier.

        Warnings and best practices

        This protocol is intended for relatively small particle boxes typical
        of 2D classification. If your images are large, consider binning or
        resizing before running it; biologically, you rarely lose anything
        important for 2D cleaning by doing so, and you gain a lot of stability
        and speed. Also, because the output evolves during streaming, always
        interpret early results as provisional; the more useful decisions
        typically come once several rounds have accumulated enough particles.
        """

    _label = 'alignPCA-2D'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_pyTorch'
    _devStatus = BETA

    # Mode
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1

    _possibleOutputs = {OUTPUT_CLASSES: SetOfClasses2D,
                        OUTPUT_AVERAGES: SetOfAverages}

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addHidden(GPU_LIST, StringParam, default='0',
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam,
                      label="Input images",
                      important=True, pointerClass='SetOfParticles',
                      help='Select the input images to be classified.')
        form.addParam('mode', EnumParam, choices=['create_classes', 'update_classes'],
                      label="Create or update 2D classes?", default=self.CREATE_CLASSES,
                      display=EnumParam.DISPLAY_HLIST,
                      help='This option allows for the classification '
                           'or simply alignment of particles into previously created classes.')
        form.addParam('numberOfClasses', IntParam, default=50,
                      condition="not mode",
                      label='Number of classes:',
                      help='Number of classes (or references) to be generated.')
        form.addParam('initialClasses', PointerParam,
                      label="Initial classes",
                      condition="mode",
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Set of initial classes to start the classification')
        form.addParam('correctCtf', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Correct CTF?',
                      help='If you set to *Yes*, the CTF of the experimental particles will be corrected')
        form.addParam('mask', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Use Gaussian Mask?',
                      help='If you set to *Yes*, a gaussian mask is applied to the images.')
        form.addParam('sigma', IntParam, default=-1, expertLevel=LEVEL_ADVANCED,
                      label='sigma:', condition="mask",
                      help='Sigma is the parameter that controls the dispersion or "width" of the curve.'
                           ' If the parameter is set to -1, sigma = dim/3.')

        form.addSection(label='Pca training')
        form.addParam('resolution', FloatParam, label="max resolution", default=8,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef', FloatParam, label="% variance", default=0.85, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of PCA components (between 0-1).'
                           ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training', IntParam, default=100000, expertLevel=LEVEL_ADVANCED,
                      label="particles for training",
                      help='Number of particles for PCA training')
        form.addSection(label='Classification')
        form.addParam('classificationBatch', IntParam, default=75000,
                      label="particles for initial classification in streaming",
                      help='Number of particles for an initial classification to compute the 2D references in streaming')

        form.addSection(label='Compute')
        form.addParam('classificationMPIs', IntParam, default=4,
                      label="MPIs",
                      help= 'MPI is used to parallelize CTF correction.' 
                            ' The CTF is corrected using the xmipp_wiener_2d function.' 
                            ' If multiple processors are available, it is recommended'
                            ' to set this value as high as possible (e.g., 24, 32...).')

        form.addParallelSection(threads=3, mpi=4)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        self._initialStep()
        self.newDeps = []
        newParticlesSet = self._loadEmptyParticleSet()

        isResume = getattr(self, '_originalRunMode', self.getRunMode()) == MODE_RESUME
        if isResume:
            self.info('Continue protocol')
            self._updateVarsToContinue()

        checkInterval = 5
        terminalStallLimit = 12
        terminalStallChecks = 0
        terminalStallSignature = None

        while not self.finish:
            # A failed step makes the executor stop and then join every
            # thread, this generator included: keep polling and the
            # run hangs for good with nothing left to do.
            if self._streamingMustStop():
                break

            # A batch may have become full while the previous
            # classification round was still running. Launch it as soon as
            # that round releases the slot, even if no new input arrives.
            if len(newParticlesSet) and self._doClassification(newParticlesSet):
                self._insertClassificationSteps(newParticlesSet, self.lastInputId)
                newParticlesSet = self._loadEmptyParticleSet()

            batchRemaining = self.classificationBatch.get() - len(newParticlesSet)
            if batchRemaining <= 0:
                # Never pass limit=0 to Set.iterItems(): Classic SQLite
                # treats a falsy limit as no LIMIT, which would consume the
                # rest of the stream into an already full pending batch.
                time.sleep(checkInterval)
                sys.stdout.flush()
                continue

            particlesSet = self._loadInputParticleSet()
            self.streamState = particlesSet.getStreamState()

            where = None
            if self.lastInputId:
                where = 'id > %d' % self.lastInputId
            tmp = None
            newCount = 0

            for particle in particlesSet.iterItems(orderBy='id',
                                                   direction='ASC',
                                                   where=where,
                                                   limit=batchRemaining):
                tmp = particle.getObjId()

                # Continue can legitimately restart from a watermark that is
                # behind particles already present in the durable output
                # (for example when publication committed but its step failed
                # afterwards). Never enqueue such particles again.
                if tmp in self._seenParticleIds:
                    continue

                newParticlesSet.append(particle.clone())
                self._seenParticleIds.add(tmp)
                newCount += 1

            inputExhausted = newCount < batchRemaining

            # Watermark discovery is the cheap hot path, but it is not
            # sufficient at end-of-stream: a lower id can become visible
            # after a higher id has already advanced lastInputId. Once the
            # producer closes, reconcile the exact logical id set and fold
            # any missing ids into the remaining room of this batch.
            if self.streamState == Set.STREAM_CLOSED:
                lateIds, terminalConsistent = self._reconcileClosedStreamIds(
                    particlesSet,
                    [],
                    self._seenParticleIds,
                    True,
                    watermarkAttr='lastInputId',
                )

                remaining = (
                    self.classificationBatch.get() - len(newParticlesSet)
                )
                if remaining > 0 and lateIds:
                    lateParticles = self._loadLogicalSetItemsByIds(
                        particlesSet,
                        lateIds[:remaining],
                    )
                    for particle in lateParticles:
                        particleId = particle.getObjId()
                        if particleId in self._seenParticleIds:
                            continue
                        newParticlesSet.append(particle)
                        self._seenParticleIds.add(particleId)
                        newCount += 1

                terminalPending = [
                    particleId
                    for particleId in lateIds
                    if particleId not in self._seenParticleIds
                ]
                inputExhausted = (
                    terminalConsistent and not terminalPending
                )

                if terminalConsistent:
                    terminalStallChecks = 0
                    terminalStallSignature = None
                elif self.classificationLaunch:
                    # A round that is still running or publishing is valid
                    # progress outside the input Set itself. Do not age the
                    # terminal inconsistency guard while that work is active.
                    terminalStallChecks = 0
                    terminalStallSignature = None
                else:
                    # A closed producer should eventually expose a complete
                    # logical Set. Allow transient reconciliation windows,
                    # but never poll forever when the durable view stops
                    # making progress.
                    stallSignature = (
                        particlesSet.getSize(),
                        len(self._seenParticleIds),
                        self.lastInputId,
                    )
                    if stallSignature != terminalStallSignature:
                        terminalStallSignature = stallSignature
                        terminalStallChecks = 1
                    else:
                        terminalStallChecks += 1

                    if terminalStallChecks >= terminalStallLimit:
                        particlesSet.close()
                        raise RuntimeError(
                            "closed input did not reach a consistent "
                            "terminal state after %d consecutive checks "
                            "without progress"
                            % terminalStallChecks
                        )

            particlesSet.close()

            if tmp is not None:
                self.lastInputId = max(self.lastInputId, tmp)

            if newCount == 0:
                self.info('No new particles')
            else:
                self.info('%d new particles in batch' % len(newParticlesSet))
                self.info('Last input id REGISTER %s' % self.lastInputId)

                if self._doClassification(newParticlesSet):
                    self._insertClassificationSteps(newParticlesSet, self.lastInputId)
                    newParticlesSet = self._loadEmptyParticleSet()

            if self.streamState == Set.STREAM_CLOSED:
                self.info('Stream closed')
                if len(newParticlesSet):
                    self.info('Finish processing with last batch %d' % len(newParticlesSet))
                    self.lastRound = True

                    # Force one more iteration to flush last batch
                    if self._doClassification(newParticlesSet):
                        self._insertClassificationSteps(newParticlesSet, self.lastInputId)
                        newParticlesSet = self._loadEmptyParticleSet()
                elif inputExhausted:
                    self._insertFunctionStep(self.closeOutputStep,
                                             prerequisites=self.newDeps,
                                             needsGPU=False)
                    self.finish = True
                    continue

            with self._lock:  # Add this lock so it will not block the iterItems of the classify method
                self.inputParticles.get().close()  # If this is not close then it blocks the input protocol
            time.sleep(checkInterval)
            sys.stdout.flush()

        sys.stdout.flush()

    # --------------------------- STEPS functions -------------------------------
    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.pcaLaunch = False
        self.classificationLaunch = False
        self.classificationRound = 0
        self.classificationStarted = False
        self.firstTimeDone = False
        self._sourceRelationPublished = False
        self.staticRun = False
        # Initialize files
        self._initFnStep()

    def _initFnStep(self):
        updateEnviron(self.gpuList.get())
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')  # Wiener
        self.imgsFn = self._getTmpPath('images_.mrc') # Wiener
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getExtraPath('classes.mrcs')
        self.sigmaProt = self.sigma.get()
        if self.sigmaProt == -1:
            self.sigmaProt = self.inputParticles.get().getDimensions()[0] / 3

        self.sampling = self.inputParticles.get().getSamplingRate()
        resolution = self.resolution.get()
        if resolution < 2 * self.sampling:
            resolution = (2 * self.sampling) + 0.5
        self.resolutionPca = resolution

        if self.mode.get() == self.UPDATE_CLASSES:
            self.numberClasses = len(self.initialClasses.get())
        else:
            self.numberClasses = self.numberOfClasses.get()

    def _insertClassificationSteps(self, newParticlesSet, lastInputId):
        # A classification round writes to fixed (non round-versioned)
        # external files. Keep only one round active at a time.
        self.classificationLaunch = True
        self._updateFnClassification()

        imgsOrigXmd = self.imgsOrigXmd
        imgsFn = self.imgsFn
        batchIds = [
            particle.getObjId()
            for particle in newParticlesSet
        ]

        # Persist only logical, serializable identity in dynamic steps.
        classStep = self._insertFunctionStep(
            self.runClassificationSteps,
            batchIds,
            imgsOrigXmd,
            imgsFn,
            prerequisites=[],
            needsGPU=True,
        )
        updateStep = self._insertFunctionStep(
            self.updateOutputSetOfClasses,
            lastInputId,
            Set.STREAM_OPEN,
            batchIds,
            prerequisites=classStep,
            needsGPU=False,
        )
        self.newDeps.append(updateStep)

    def runClassificationSteps(self, particleIds, imgsOrigXmd, imgsFn):
        """Run one round from serializable logical particle ids."""
        requestedIds = [
            int(particleId)
            for particleId in particleIds
        ]
        requestedIdSet = set(requestedIds)

        inputSet = self._loadInputParticleSet()
        try:
            alignment = inputSet.getAlignment()
            particles = self._loadLogicalSetItemsByIds(
                inputSet,
                requestedIds,
            )
        finally:
            inputSet.close()

        loadedIds = {
            particle.getObjId()
            for particle in particles
        }
        if loadedIds != requestedIdSet:
            missingIds = sorted(requestedIdSet - loadedIds)
            unexpectedIds = sorted(loadedIds - requestedIdSet)
            raise RuntimeError(
                "Could not reconstruct the complete AlignPCA "
                "classification batch from the logical input. "
                "Missing ids: %s. Unexpected ids: %s."
                % (missingIds, unexpectedIds)
            )

        batchView = _LogicalParticleBatchView(
            particles,
            alignment,
        )
        self.convertInputStep(batchView, imgsOrigXmd, imgsFn)

        numTrain = min(len(batchView), self.training.get())
        self.classification(
            imgsFn,
            self.numberClasses,
            imgsOrigXmd,
            self.mask.get(),
            self.sigmaProt,
            numTrain,
            self.resolutionPca,
        )

        # Keep retry evidence on scientific failure. Clean only after the
        # external classification has returned successfully.
        pwutils.cleanPath(imgsFn)

    def convertInputStep(self, input, outputOrig, outputMRC):
        referenceMustExist = (
            self.classificationStarted
            or (
                self.mode.get() == self.UPDATE_CLASSES
                and self.firstTimeDone
            )
        )
        if referenceMustExist and not os.path.exists(self.ref):
            raise RuntimeError(
                "Classification reference is missing after durable "
                "classification state was restored: %s"
                % self.ref
            )

        writeSetOfParticles(input, outputOrig)

        if self.correctCtf.get():
            # ------------ WIENER -----------------------
            args = ' -i %s  --operate  randomize'%(outputOrig)
            self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)
            
            args = ' -i  %s -o %s --sampling_rate %s '%(outputOrig, outputMRC, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get())
            
        else:
            args = ' -i  %s -o %s  ' % (outputOrig, outputMRC)
            self.runJob("xmipp_image_convert", args)

        # For classification update
        if self.mode.get() == self.UPDATE_CLASSES and not self.firstTimeDone:
            initial2DClasses = self.initialClasses.get()
            self.info('Write classes')
            if isinstance(initial2DClasses, SetOfClasses2D):
                writeSetOfClasses2D(initial2DClasses,
                                    self.refXmd, writeParticles=False)
            else:
                writeSetOfParticles(initial2DClasses,
                                    self.refXmd,
                                    alignType=ALIGN_NONE)

            args = ' -i  %s -o %s ' % (self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args)
            self.firstTimeDone = True

    def classification(self, inputIm, numClass, stfile, mask, sigma, numTrain, resolutionTrain):
        args = ' -i %s -s %s -c %s -t %s -hr %s -p %s  -o %s/classes -stExp %s' % \
               (inputIm, self.sampling, numClass, numTrain, resolutionTrain, self.coef.get(), self._getExtraPath(),
                stfile)
        if mask:
            args += ' --mask --sigma %s ' % (sigma)

        if self.mode.get() == self.UPDATE_CLASSES or self._isClassificationDone():
            args += ' -r %s ' % self.ref

        env = self.getCondaEnv()
        env = self._setEnvVariables(env)
        self.runJob("xmipp_alignPCA_2D", args, env=env)
        
        args = ' -i %s  --operate  sort itemId'%(self._getExtraPath(AVERAGES_IMAGES_FILE))
        self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)

    def updateOutputSetOfClasses(self, lastInputId, streamMode,
                                   batchIds=None):
        outputName = OUTPUT_CLASSES
        outputClasses, update = self._loadOutputSet(outputName)

        self._fillClassesFromLevel(
            outputClasses,
            update,
            batchIds=batchIds,
        )
        self._updateOutputSet(outputName, outputClasses, streamMode)
        # self._updateOutputAverages(update)

        if not update:  # First time
            XmippProtClassifyPcaStreaming._ensureSourceRelation(
                self,
                outputClasses,
            )
            self.classificationStarted = True
            self.numberClasses = len(outputClasses)  # In case the original number of classes is not reached
        elif getattr(self, '_sourceRelationPublished', None) is False:
            XmippProtClassifyPcaStreaming._ensureSourceRelation(
                self,
                outputClasses,
            )

        self.lastInputIdProcessed = lastInputId
        self.info(r'Last input id processed UPDATED is %s' % str(self.lastInputIdProcessed))
        self.info(r'Last classification round processed is %d' % self.classificationRound)

        # Durable truth for Continue/Resume is this step's own persisted
        # record (funcName + args) in the step graph, reconstructed by
        # _restoreStreamingStateFromSteps - not a last_done.txt/
        # classification_done.txt sidecar. classificationRound already
        # advanced exactly once for this round, inside
        # _updateFnClassification (incrementing it again here would just
        # make every other round number unused, with no effect on
        # uniqueness - _updateFnClassification's own increment already
        # guarantees the next round gets a fresh file-name suffix).

        # This round's external files (classes_classes.star,
        # classes_images.star) have now been fully read - it is safe for
        # stepsGeneratorStep to launch the next round, which will
        # overwrite those same fixed filenames.
        self.classificationLaunch = False

    def closeOutputStep(self):
        self._closeOutputSet()

    # --------------------------- UTILS functions -----------------------------
    def _loadInputParticleSet(self):
        """Return the logical input particle Set with refreshed state."""
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()

        return partSet

    def _ensureSourceRelation(self, outputClasses):
        """Ensure that the durable output keeps its source provenance."""
        sourcePointer = self._getInputPointer()
        relationExists = False

        mapper = getattr(self, 'mapper', None)
        getRelationParents = getattr(
            mapper, 'getRelationParents', None
        )

        if callable(getRelationParents):
            sourceObject = sourcePointer
            isPointer = getattr(sourcePointer, 'isPointer', None)
            if callable(isPointer) and isPointer():
                sourceObject = sourcePointer.getObjValue()

            sourceId = sourceObject.getObjId()
            relationExists = any(
                parent.getObjId() == sourceId
                for parent in getRelationParents(
                    RELATION_SOURCE,
                    outputClasses,
                )
            )

        if not relationExists:
            self._defineSourceRelation(
                sourcePointer,
                outputClasses,
            )

        self._sourceRelationPublished = True

    def _getInputPointer(self):
        return self.inputParticles

    def _loadEmptyParticleSet(self):
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()
        copyPartSet = self._createSetOfParticles()
        copyPartSet.copyInfo(partSet)

        return copyPartSet

    def _setEnvVariables(self, env):
        """ Method to set all the environment variables needed to run PCA program """
        env['LD_LIBRARY_PATH'] = ''
        # Limit the number of threads
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        return env

    def _updateFnClassification(self):
        """Update input based on the iteration it is"""
        self.imgsOrigXmd = updateFileName(self.imgsOrigXmd, self.classificationRound)
        self.imgsXmd = updateFileName(self.imgsXmd, self.classificationRound)
        self.imgsFn = updateFileName(self.imgsFn, self.classificationRound)
        self.info('Starts classification round: %d' % self.classificationRound)
        self.classificationRound += 1

    def _classifyLogicalParticles(self, clsSet, particles, mdIter):
        """Classify one exact logical particle batch into the output classes."""
        clsDict = clsSet._getExistingItems()

        for particle in particles:
            if particle.isEnabled():
                row = next(mdIter)
                self._updateParticle(particle, row)

                if not getattr(particle, "_appendItem", True):
                    continue

                classId = particle.getClassId()
                if classId is None:
                    raise RuntimeError(
                        "Particle %s has no class assignment."
                        % particle.getObjId()
                    )
                if classId == 0:
                    continue

                classItem = clsSet._get_or_create_class(
                    clsDict,
                    classId,
                    self._updateClass,
                )
                classItem.append(particle)
            else:
                next(mdIter)

        for classItem in clsDict.values():
            clsSet.update(classItem)

    def _fillClassesFromLevel(self, clsSet, update=False, batchIds=None):
        """Create or update the output classes from one logical batch."""
        self._createModelFile()
        self._loadClassesInfo(
            self._getExtraPath(CONTRAST_AVERAGES_FILE)
        )

        images = clsSet.getImages()

        if batchIds is None:
            visibleIds = images.getIdSet()
            if update:
                particleIds = sorted(
                    particleId
                    for particleId in visibleIds
                    if particleId > self.lastInputIdProcessed
                )
            else:
                particleIds = sorted(visibleIds)
        else:
            particleIds = sorted({
                int(particleId)
                for particleId in batchIds
            })

        particles = (
            self._loadLogicalSetItemsByIds(images, particleIds)
            if particleIds
            else []
        )

        particlesById = {
            particle.getObjId(): particle
            for particle in particles
        }
        loadedIds = set(particlesById)
        expectedIds = set(particleIds)

        if (
                loadedIds != expectedIds
                or len(particlesById) != len(particles)
        ):
            missingIds = sorted(expectedIds - loadedIds)
            unexpectedIds = sorted(loadedIds - expectedIds)
            raise RuntimeError(
                "Could not reconstruct the exact logical publication batch. "
                "Missing ids: %s. Unexpected ids: %s."
                % (missingIds, unexpectedIds)
            )

        orderedParticles = [
            particlesById[particleId]
            for particleId in particleIds
        ]

        mdRows = emtable.Table.iterRows(
            'particles@' + self._getExtraPath(AVERAGES_IMAGES_FILE)
        )
        mdIter = self._iterRowsForParticleIds(
            mdRows,
            particleIds,
        )

        with self._lock:
            self._classifyLogicalParticles(
                clsSet,
                orderedParticles,
                mdIter,
            )

    @staticmethod
    def _iterRowsForParticleIds(rows, particleIds):
        rowIter = iter(rows)
        row = next(rowIter, None)

        for particleId in particleIds:
            while (
                    row is not None
                    and row.get(XMIPPCOLUMNS.itemId.value) < particleId
            ):
                row = next(rowIter, None)

            if (
                    row is not None
                    and row.get(XMIPPCOLUMNS.itemId.value) == particleId
            ):
                yield row
                row = next(rowIter, None)
            else:
                yield None

    def _loadOutputSet(self, outputName):
        """
        Load the output set if it exists or create a new one.
        """
        outputSet = getattr(self, outputName, None)
        update = False
        if outputSet is None:
            outputSet = self._createSetOfClasses2D(self._getInputPointer())
            outputSet.setStreamState(Set.STREAM_OPEN)
        else:
            outputSet.loadAllProperties()
            update = True

        return outputSet, update

    def _doClassification(self, newParticlesSet):
        """ Three cases for launching Classification
            - First round of classification: enough particles and PCA done
            - Update classification: classification done, PCA done and enough batch size
            - First or update classification with a smaller batch: last round of particles and not classification launch
        """
        n = len(newParticlesSet)
        b = self.classificationBatch.get()
        if self.classificationLaunch:
            return False

        batchReady = n >= b
        lastRoundReady = self.lastRound

        return batchReady or lastRoundReady


    def _isClassificationDone(self):
        return self.classificationStarted or self.mode.get() == self.UPDATE_CLASSES

    def _restoreStreamingStateFromSteps(self):
        """ Reconstruct lastInputId/classificationRound/classificationStarted
        from the persisted step graph instead of last_done.txt/
        classification_done.txt sidecars - the step graph pyworkflow
        already maintains durably is the source of truth, same principle
        as XmippStreamingBase's own _isFunctionStepFinished for other
        protocols. Only FINISHED updateOutputSetOfClasses steps count:
        an interrupted round's particles are simply rediscovered as new
        input on the next check (self.lastInputId stays at the last
        completed round's value) and reprocessed into a fresh round,
        safely overwriting that round's half-written fixed filenames.
        """
        lastInputId = 0
        finishedRounds = 0

        for step in self._iterKnownStreamingSteps():
            funcName = getattr(step, "funcName", None)
            if hasattr(funcName, "get"):
                funcName = funcName.get()
            if funcName != "updateOutputSetOfClasses" or not step.isFinished():
                continue

            argsStr = getattr(step, "argsStr", None)
            if hasattr(argsStr, "get"):
                argsStr = argsStr.get("[]")
            try:
                args = json.loads(argsStr or "[]")
            except (TypeError, ValueError):
                continue
            if not args:
                continue

            finishedRounds += 1
            try:
                stepLastInputId = int(args[0])
            except (TypeError, ValueError, IndexError):
                continue
            lastInputId = max(lastInputId, stepLastInputId)

        return lastInputId, finishedRounds

    def _restoreProcessedParticleIdsFromSteps(self):
        """Return exact particle ids owned by durable publication steps.

        New AlignPCA publication steps persist the batch ids as their third
        argument. Older runs did not, so return ``exact=False`` when at
        least one FINISHED publication step lacks that information; the
        caller then falls back to the legacy watermark semantics for those
        already-finished rounds.
        """
        processedIds = set()
        exact = True

        for step in self._iterKnownStreamingSteps():
            funcName = getattr(step, 'funcName', None)
            if hasattr(funcName, 'get'):
                funcName = funcName.get()
            if (
                    funcName != 'updateOutputSetOfClasses'
                    or not step.isFinished()
            ):
                continue

            argsStr = getattr(step, 'argsStr', None)
            if hasattr(argsStr, 'get'):
                argsStr = argsStr.get('[]')
            try:
                args = json.loads(argsStr or '[]')
            except (TypeError, ValueError):
                exact = False
                continue

            if len(args) < 3 or not isinstance(args[2], list):
                exact = False
                continue

            for particleId in args[2]:
                try:
                    processedIds.add(int(particleId))
                except (TypeError, ValueError):
                    exact = False

        return processedIds, exact

    def _updateVarsToContinue(self):
        """ If the protocol is set to continue, reconstruct where it was
        stopped from the persisted step graph. """
        lastInputId, finishedRounds = self._restoreStreamingStateFromSteps()

        self.lastInputId = lastInputId
        # _updateFnClassification advances classificationRound exactly
        # once per round (before that round even runs) - after N rounds
        # have finished, the in-memory counter sits at N, so resuming
        # with classificationRound = finishedRounds reproduces the same
        # value and the next round gets a fresh, never-before-used
        # file-name suffix.
        self.classificationRound = finishedRounds
        self.classificationStarted = finishedRounds > 0
        if finishedRounds > 0:
            if self.mode.get() == self.UPDATE_CLASSES:
                self.firstTimeDone = True
            else:
                # In CREATE_CLASSES the first classification round may
                # produce fewer classes than the user originally requested.
                # Continue must therefore restore the effective class count
                # from the durable output, not from numberOfClasses.
                outputClasses = getattr(self, OUTPUT_CLASSES, None)
                if outputClasses is not None:
                    loadAllProperties = getattr(
                        outputClasses, 'loadAllProperties', None
                    )
                    if callable(loadAllProperties):
                        loadAllProperties()

                    getSize = getattr(outputClasses, 'getSize', None)
                    persistedClassCount = (
                        getSize() if callable(getSize)
                        else len(outputClasses)
                    )
                    if persistedClassCount:
                        self.numberClasses = persistedClassCount

        processedIds = set()
        exactIds = False
        restoreIds = getattr(
            self, '_restoreProcessedParticleIdsFromSteps', None
        )
        if callable(restoreIds):
            processedIds, exactIds = restoreIds()
            self._seenParticleIds = set(processedIds)

            # Compatibility with runs created before exact batch ids were
            # persisted. Preserve their historical watermark semantics,
            # while new runs remain hole-safe.
            if finishedRounds > 0 and not exactIds:
                inputSet = self._loadInputParticleSet()
                try:
                    try:
                        visibleIds = inputSet.getUniqueValues('id')
                    except NotImplementedError:
                        visibleIds = [
                            particle.getObjId()
                            for particle in inputSet
                        ]
                    self._seenParticleIds.update(
                        int(particleId)
                        for particleId in visibleIds
                        if int(particleId) <= self.lastInputId
                    )
                finally:
                    inputSet.close()

        # Step status is not enough after an ambiguous commit: a publication
        # step may fail after outputClasses has already been written. Durable
        # class membership is therefore authoritative on Continue.
        outputClasses = getattr(self, OUTPUT_CLASSES, None)
        durableIds = set()
        persistedClassCount = 0

        if outputClasses is not None:
            loadAllProperties = getattr(
                outputClasses, 'loadAllProperties', None
            )
            if callable(loadAllProperties):
                loadAllProperties()

            getSize = getattr(outputClasses, 'getSize', None)
            persistedClassCount = (
                getSize() if callable(getSize)
                else len(outputClasses)
            )

            iterClassItems = getattr(outputClasses, 'iterClassItems', None)
            if callable(iterClassItems):
                durableIds = {
                    particle.getObjId()
                    for particle in iterClassItems()
                    if particle.getObjId() is not None
                }
                self._seenParticleIds.update(durableIds)

        if durableIds:
            # Durable output also implies that its provenance must exist.
            # Re-check it on Continue so an interrupted first publication
            # cannot leave a permanent provenance gap.
            if getattr(self, 'mapper', None) is not None:
                XmippProtClassifyPcaStreaming._ensureSourceRelation(
                    self,
                    outputClasses,
                )

            # A non-empty durable output proves that the initial
            # classification exists, even if its publication step failed
            # after committing the output.
            self.classificationStarted = True

            if self.mode.get() == self.CREATE_CLASSES:
                if persistedClassCount:
                    self.numberClasses = persistedClassCount
            elif self.mode.get() == self.UPDATE_CLASSES:
                self.firstTimeDone = True

            # With exact ids from FINISHED publication steps, any durable
            # particle not owned by those steps proves one additional
            # ambiguous-but-committed round. Only one round can be in flight
            # because classificationLaunch serializes publication.
            if exactIds and durableIds.difference(processedIds):
                self.classificationRound = max(
                    self.classificationRound,
                    finishedRounds + 1,
                )
            elif (
                    not exactIds
                    and any(
                        int(particleId) > self.lastInputId
                        for particleId in durableIds
                    )
            ):
                # Older persisted steps may not identify their exact batch.
                # Durable ids beyond the last completed watermark prove that
                # one additional serialized publication round committed.
                self.classificationRound = max(
                    self.classificationRound,
                    finishedRounds + 1,
                )

        self.lastInputIdProcessed = self.lastInputId

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are no errors. If some errors are found, a list with
        the error messages will be returned.
        """
        errors = []
        if self.inputParticles.get().getDimensions()[0] > 256:
            errors.append("You should resize the particles."
                          " Sizes smaller than 128 pixels are recommended.")
        er = self.validateDLtoolkit()
        if not isinstance(er, list):
            er = [er]
        if er:
            errors+=er
        return errors
    
    def _warnings(self):
        validateMsgs = []
        if self.inputParticles.get().getDimensions()[0] > 128:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        if self.inputParticles.get().getDimensions()[0] > 256:
            validateMsgs.append("Particle sizes bigger than 256 may"
                                " saturate the GPU memory.")
        return validateMsgs
    
    def _summary(self):
        summary = []

        if not hasattr(self, 'outputClasses'):
            summary.append("Output classes not ready yet.")
        else:
            summary.append('2D clasification using AlignPCA')

        return summary

    # #--------------------------- UTILS functions -------------------------------
    # EMTABLE IMPLEMENTATION
    def _updateParticle(self, item, row):
        particleId = item.getObjId()

        if row is None:
            raise RuntimeError(
                "Missing AlignPCA classification metadata row for "
                "particle %s." % particleId
            )

        rowParticleId = row.get(XMIPPCOLUMNS.itemId.value)
        if particleId != rowParticleId:
            raise RuntimeError(
                "AlignPCA classification metadata is not synchronized: "
                "particle %s received metadata for particle %s."
                % (particleId, rowParticleId)
            )

        item.setClassId(row.get(XMIPPCOLUMNS.ref.value))
        item.setTransform(rowToAlignmentEmtable(row, ALIGN_2D))

    def _updateClass(self, item):
        classId = item.getObjId()
        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())

    def _createModelFile(self):
        with open(self._getExtraPath(CONTRAST_AVERAGES_FILE), 'r') as file:
            # Read the lines of the file
            lines = file.readlines()
        # Open the file for writing
        with open(self._getExtraPath(CONTRAST_AVERAGES_FILE), "w") as file:
            # Iterate through the lines
            for line in lines:
                # Replace "data_" with "data_particles" if found
                modifiedLine = line.replace("data_", "data_particles")
                # Write the modified line to the file
                file.write(modifiedLine)

        with open(self._getExtraPath(AVERAGES_IMAGES_FILE), 'r') as file:
            lines = file.readlines()

            # Find the index of the last non-empty line
        lastNonEmptyInd = len(lines) - 1
        while lastNonEmptyInd >= 0 and lines[lastNonEmptyInd].strip() == "":
            lastNonEmptyInd -= 1

        # Modify the lines
        modifiedLines = []
        for line in lines[:lastNonEmptyInd + 1]:
            # Replace "data_" with "data_particles" if found
            # modifiedLine = line.replace("data_Particles", "data_particles")
            modifiedLine = line.replace("data_noname", "data_particles")
            modifiedLines.append(modifiedLine)

        # Write the modified lines back to the file
        with open(self._getExtraPath(AVERAGES_IMAGES_FILE), "w") as file:
            file.writelines(modifiedLines)

    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id

        mdFileName = '%s@%s' % ('particles', filename)
        table = emtable.Table(fileName=filename)

        for classNumber, row in enumerate(table.iterRows(mdFileName)):
            index, fn = xmippToLocation(row.get(XMIPPCOLUMNS.image.value))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[classNumber + 1] = (index, fn, row)
        self._numClass = index

# --------------------------- Static functions --------------------------------
def rowToAlignmentEmtable(alignmentRow, alignType):
    """
    is2D == True-> matrix is 2D (2D images alignment)
            otherwise matrix is 3D (3D volume alignment or projection)
    invTransform == True  -> for xmipp implies projection
    """

    is2D = alignType == ALIGN_2D
    inverseTransform = alignType == ALIGN_PROJ

    if alignmentRow.hasAnyColumn(ALIGNMENT_DICT.values()):
        alignment = Transform()
        angles = np.zeros(3)
        shifts = np.zeros(3)
        flip = alignmentRow.get(XMIPPCOLUMNS.flip.value, default=0.)

        shifts[0] = alignmentRow.get(XMIPPCOLUMNS.shiftX.value, default=0.)
        shifts[1] = alignmentRow.get(XMIPPCOLUMNS.shiftY.value, default=0.)

        if not is2D:
            angles[0] = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            angles[1] = alignmentRow.get(XMIPPCOLUMNS.angleTilt.value, default=0.)
            angles[2] = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            shifts[2] = alignmentRow.get(XMIPPCOLUMNS.shiftZ.value, default=0.)
            if flip:
                angles[1] = angles[1] + 180  # tilt + 180
                angles[2] = - angles[2]  # - psi, COSS: this is mirroring X
                shifts[0] = - shifts[0]  # -x
        else:
            psi = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            rot = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            if not np.isclose(rot, 0., atol=1e-6) and not np.isclose(psi, 0., atol=1e-6):
                print("HORROR rot and psi are different from zero in 2D case")

            angles[0] = psi + rot

        M = matrixFromGeometry(shifts, angles, inverseTransform)

        if flip:
            if alignType == ALIGN_2D:
                M[0, :2] *= -1.  # invert only the first two columns
                # keep x
                M[2, 2] = -1.  # set 3D rot
            elif alignType == ALIGN_3D:
                M[0, :3] *= -1.  # now, invert first line excluding x
                M[3, 3] *= -1.
            elif alignType == ALIGN_PROJ:
                pass

        alignment.setMatrix(M)

    else:
        alignment = None

    return alignment


# --------------------------- Static functions --------------------------------
def updateFileName(filepath, classRound):
    filename = os.path.basename(filepath)
    newFilename = f"{filename[:filename.find('_')]}_{classRound}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), newFilename)