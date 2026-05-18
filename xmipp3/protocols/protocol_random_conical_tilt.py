# **************************************************************************
# *
# * Authors:     Javier Vargas (jvargas@cnb.csic.es)
# *              Adrian Quintana (aquintana@cnb.csic.es)
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

from os.path import exists

from pyworkflow.object import String
from pyworkflow.protocol.constants import LEVEL_ADVANCED
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import cleanPath

from pwem.protocols import ProtInitialVolume
from pwem.objects import Volume, SetOfParticles
from pwem.constants import ALIGN_2D
from pwem import emlib

from xmipp3.convert import getImageLocation, alignmentToRow, convertToMrc


class XmippProtRCT(ProtInitialVolume):
    """Creates initial 3D volumes using 2D classes and particle pairs from
    tilted images. Applies the Random Conical Tilt (RCT) method to generate
    unbiased starting volume for structure refinement, even with a small number
    of image pairs. The volume serves as starting models for further
    refinement steps.

    AI Generated

    ## Overview

    The Random Conical Tilt protocol creates one or more initial 3D volumes from
    tilted-pair particle data using the Random Conical Tilt, or RCT, method.

    RCT is a classical strategy for obtaining initial 3D models from pairs of
    untilted and tilted images. The untilted particles are first aligned or grouped
    into 2D classes. The corresponding tilted particles are then used, together
    with the known tilt geometry, to reconstruct a 3D volume.

    This protocol is useful when the user has acquired tilted-pair data and wants
    to generate starting models for later 3D refinement. Because the geometry of
    the tilted acquisition provides important orientation information, RCT can
    produce initial volumes even when no previous 3D reference is available.

    The output is a set of reconstructed initial volumes. Optionally, the protocol
    also produces low-pass-filtered versions of those volumes.

    ## Inputs and General Workflow

    The protocol requires two related inputs:

    - a set of tilted-pair particles;
    - a set of input particles or 2D classes with alignment information.

    The tilted-pair particle set provides the relationship between each untilted
    particle and its tilted counterpart, together with the tilt-angle information.
    The input particles or classes define the aligned untilted views that will be
    used to organize the reconstruction.

    The protocol first creates Xmipp metadata linking each untilted particle,
    tilted particle, 2D alignment, micrograph pair, and tilt angle. Then, for each
    input class or particle group, it aligns the tilted particles with respect to
    the corresponding untilted reference and reconstructs a 3D volume using ART
    reconstruction.

    If filtering is enabled, each reconstructed volume is also low-pass filtered.

    ## Input Particles Tilt Pair

    The **Input particles tilt pair** parameter should point to a
    **ParticlesTiltPair** object.

    This object contains two linked particle sets:

    - the untilted particles;
    - the tilted particles.

    It also contains the coordinate-pair and micrograph-pair information needed to
    associate each particle with its corresponding tilted view. The tilt angles are
    read from the paired-coordinate metadata.

    This input is the geometrical foundation of the protocol. If the tilted and
    untilted particles are not correctly paired, the reconstructed RCT volumes will
    not be reliable.

    ## Input Classes or Particles

    The **Input classes** parameter can be either a set of particles or a set of 2D
    classes.

    These input images must contain 2D alignment information. The protocol checks
    this requirement before running.

    If a **SetOfParticles** is provided, the protocol reconstructs one RCT volume
    from that particle set.

    If a **SetOfClasses** is provided, the protocol reconstructs one RCT volume for
    each class. This is a common use case: untilted particles are first classified
    into 2D classes, and then each class is used to generate a separate initial 3D
    volume from the corresponding tilted particles.

    Using classes can help separate different views, conformations, or particle
    subsets before RCT reconstruction.

    ## 2D Alignment Requirement

    The input particles or classes must have alignment information.

    This alignment describes how the untilted particles are positioned and rotated
    within the 2D class or particle set. The protocol uses this information when
    creating the RCT metadata and when relating untilted and tilted images.

    If the input particles or classes are not aligned, the RCT reconstruction will
    not have a meaningful orientation framework.

    In practice, users should usually perform 2D alignment or 2D classification on
    the untilted particles before running this protocol.

    ## Creating RCT Metadata

    The protocol creates an Xmipp metadata file containing the information needed
    for RCT reconstruction.

    For each particle, the metadata includes:

    - the untilted particle image;
    - the tilted particle image;
    - the untilted and tilted micrograph names;
    - the particle coordinates;
    - the 2D alignment parameters;
    - the tilt-pair angles;
    - the particle identifier.

    This metadata is created separately for each input class or particle group.

    The protocol also accounts for possible differences in sampling rate between
    the tilted-pair particles and the input aligned particles by scaling the 2D
    alignment parameters accordingly.

    ## Thin Object Option

    The **Thin Object** option allows the protocol to stretch tilted projections to
    better match the untilted projections.

    This option may be useful when the specimen is physically thin and the effect
    of tilting can be approximated by a stretching operation in projection.

    It should be used only when this assumption is appropriate for the specimen.
    For globular or thick particles, enabling this option may introduce an
    inappropriate deformation.

    ## Maximum Allowed Shift for Tilted Particles

    The **Maximum allowed shift for tilted particles** parameter defines the
    largest allowed shift, in pixels, during alignment of the tilted particles.

    Tilted particles whose estimated shift exceeds this value are discarded.

    This is a quality-control parameter. Very large shifts may indicate incorrect
    pairing, poor tilted-image quality, wrong alignment, or particles that cannot
    be reliably matched between tilted and untilted views.

    A larger value is more permissive and discards fewer particles. A smaller value
    is stricter and may remove problematic tilted particles, but it may also remove
    valid particles if the initial alignment is difficult.

    If the value is set larger than the image size, effectively no particles are
    discarded by this criterion.

    ## Skip Tilted Translation Alignment

    The **Skip tilted translation alignment** option disables translation alignment
    of the tilted particles.

    Normally, the protocol aligns the tilted images with respect to the untilted
    reference or class average. This helps improve the consistency of the tilted
    particles before reconstruction.

    However, if the tilted images have very low quality, translation alignment may
    produce poor estimates and may make the reconstruction worse. In such cases,
    skipping tilted translation alignment can be safer.

    This is an advanced option and should be used when the user has reason to
    suspect that tilted-particle translation alignment is unreliable.

    ## Reconstruction Parameters

    The **Additional reconstruction parameters** field allows the user to provide
    extra options for the Xmipp ART reconstruction program.

    The default parameters are intended as a reasonable starting point for RCT
    initial-volume reconstruction.

    Advanced users can modify these parameters to change reconstruction behavior,
    such as the number of iterations or regularization-related settings. However,
    these options should be changed carefully because RCT reconstructions are often
    based on limited and noisy tilted-pair data.

    Most users should keep the default reconstruction parameters unless they have a
    specific reason to tune the ART reconstruction.

    ## Filtering Reconstructed Volumes

    The **Filter reconstructed volumes?** option applies a low-pass filter to the
    reconstructed RCT volumes.

    RCT volumes are often noisy, especially when they are reconstructed from a
    small number of particle pairs or from tilted images with low signal-to-noise
    ratio. Low-pass filtering can make the initial volume easier to interpret and
    more suitable as a starting reference.

    The protocol can output both the unfiltered reconstructed volumes and the
    filtered versions.

    ## Resolution of the Low-Pass Filter

    The **Resolution of the low-pass filter** parameter defines the cutoff used for
    filtering the reconstructed volumes, expressed in digital frequency.

    Lower cutoff values apply stronger low-pass filtering. Higher values preserve
    more high-frequency information.

    For initial models, moderate low-pass filtering is usually appropriate because
    the goal is to obtain the correct overall shape and orientation, not to
    interpret high-resolution features.

    Excessively weak filtering may leave noisy artifacts in the initial volume,
    whereas excessively strong filtering may remove useful shape information.

    ## Output Volumes

    The main output is **outputVolumes**.

    This output contains one reconstructed RCT volume for each input particle group
    or 2D class. If the input is a single particle set, one volume is produced. If
    the input is a set of classes, one volume is produced per class.

    The volumes are converted to MRC format and registered in Scipion with the
    sampling rate of the untilted particles from the tilted-pair input.

    These volumes are intended mainly as initial models for further refinement.

    ## Output Filtered Volumes

    If filtering is enabled, the protocol also produces
    **outputFilteredVolumes**.

    These are low-pass-filtered versions of the RCT reconstructions. In many
    workflows, the filtered volumes are more useful as starting references because
    they suppress noise and emphasize the global shape.

    Users should inspect both filtered and unfiltered outputs when deciding which
    volume to use for downstream refinement.

    ## Interpreting RCT Volumes

    RCT volumes should be interpreted as initial models, not as final
    high-resolution reconstructions.

    They are often affected by missing information, noise, limited angular
    coverage, tilted-image quality, and the number of particles contributing to
    each class. Their main value is to provide a plausible 3D starting point for
    later refinement.

    When one volume is reconstructed per 2D class, differences between RCT volumes
    may reflect different views, conformational states, particle quality, or class
    composition. Some classes may produce better initial volumes than others.

    ## Practical Recommendations

    Before running RCT, make sure that tilted and untilted particles are correctly
    paired.

    Use 2D classes from untilted particles when possible. Class averages provide
    cleaner references for aligning the tilted particles than individual noisy
    particles.

    Use classes with enough particles. Very small classes may produce noisy or
    unstable RCT volumes.

    Keep filtering enabled for most initial-model workflows. Filtered RCT volumes
    are usually more useful for subsequent refinement.

    Inspect the tilted-particle alignment and the resulting volumes. Discarded
    particles or poor volumes may indicate problems in tilted-pair picking,
    pairing, or class quality.

    Use the maximum-shift parameter to reject tilted particles that cannot be
    aligned reliably, but avoid making it too strict at the beginning.

    Treat the output as an initial reference. It should usually be refined further
    with a standard 3D refinement protocol.

    ## Final Perspective

    Random Conical Tilt is an initial-volume reconstruction protocol for
    tilted-pair cryo-EM data.

    For biological users, its main value is that it can generate a 3D starting
    model from experimental tilt-pair geometry, without requiring an external
    reference. This is especially useful for new specimens or workflows where
    reference bias should be minimized.

    The reliability of the output depends strongly on correct tilted-pair
    coordinates, good untilted 2D alignment or classification, and sufficient
    tilted-particle quality. Used carefully, RCT provides a practical bridge from
    paired 2D observations to an initial 3D model.
    """
    _label = 'random conical tilt'
    
    def __init__(self, **args):
        ProtInitialVolume.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        # TODO: Input can be either a SetOfParticles or
        # a SetOfClasses2D
        form.addParam('inputParticlesTiltPair', params.PointerParam,
                      label="Input particles tilt pair",
                      important=True,
                      pointerClass='ParticlesTiltPair',
                      help='Select the input particles tilt pair file '
                           'that will be used.  file. This file is used '
                           'to associate each micrograph with its tilted '
                           'equivalent.')
        
        form.addParam('inputParticles', params.PointerParam,
                      label="Input classes", important=True,
                      pointerClass='SetOfParticles,SetOfClasses',
                      help='Select the input images or classes from '
                           'the project.')
        
        form.addSection(label='Alignment parameters')
        form.addParam('thinObject', params.BooleanParam,
                      default=False, label='Thin Object',
                      help='If the object is thin, then the tilted '
                           'projections can be stretched to match the '
                           'untilted projections')
                       
        form.addParam('maxShift', params.IntParam,
                      default="10", expertLevel=LEVEL_ADVANCED,
                      label="Maximum allowed shift for tilted particles (pixels)", 
                      help='Particles that shift more will be discarded. '
                           'A value larger than the image size will not '
                           'discard any particle.')
        
        form.addParam('skipTranslation', params.BooleanParam,
                      default=False, expertLevel=LEVEL_ADVANCED,
                      label='Skip tilted translation alignment', 
                      help='If the tilted image quality is very low, '
                           'then this alignment might result in poor '
                           'estimates.')

        form.addSection(label='Reconstruction')
        form.addParam('additionalParams', params.StringParam,
                      default="-n 5 -l 0.01", expertLevel=LEVEL_ADVANCED,
                      label='Additional reconstruction parameters', 
                      help='See: http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Reconstruct_art_v31')
        
        form.addParam('doFilter', params.BooleanParam, default=True,
                      label='Filter reconstructed volumes?', 
                      help='Filtering may be useful to remove noise, '
                           'especially when few particles '
                           'contribute to the reconstruction.')
        
        form.addParam('resoLowPassFilter', params.FloatParam, default=0.2,
                      label='Resolution of the low-pass filter (dig.freq)',
                      help='Resolution of the low-pass filter (dig.freq)')        

        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- STEPS functions -------------------------------

    def _insertAllSteps(self):
        self.inputSet = self.inputParticles.get()
        self.rctClassesFn = self._getExtraPath('rct_classes.xmd')
        
        firstStepId = self._insertFunctionStep('createRctImagesStep')
        reconSteps = []
                
        if isinstance(self.inputSet, SetOfParticles):
            reconStep = self._reconstructImages(self.inputSet, firstStepId)
            reconSteps.append(reconStep)
        else:
            for class2D in self.inputSet:
                reconStep = self._reconstructImages(class2D, firstStepId)
                reconSteps.append(reconStep)
                
        self._insertFunctionStep('createOutputStep', prerequisites=reconSteps)      
        
    def createRctImagesStep(self):
        """ Function to create the Xmipp metadata needed to run the
        Xmipp protocol """
        if isinstance(self.inputSet, SetOfParticles):
            self._appendRctImages(self.inputSet)
        else:
            for class2D in self.inputSet:
                self._appendRctImages(class2D)

    def _appendRctImages(self, particles):
        blockMd = "class%06d_images@%s" % (particles.getObjId(),
                                           self.rctClassesFn)
        classMd = emlib.MetaData()

        partPairs = self.inputParticlesTiltPair.get()
        uImages = partPairs.getUntilted()
        tImages = partPairs.getTilted()
        sangles = partPairs.getCoordsPair().getAngles()

        micPairs = partPairs.getCoordsPair().getMicsPair()
        uMics = micPairs.getUntilted()
        tMics = micPairs.getTilted()
        
        scaleFactor = uImages.getSamplingRate() / particles.getSamplingRate()
        
        for img in particles:
            imgId = img.getObjId()
                       
            uImg = uImages[imgId]
            tImg = tImages[imgId]
            
            if uImg is None or tImg is None:
                print(">>> Warning, for id %d, tilted or untilted particle "
                      "was not found. Ignored." % imgId)
            else:
                objId = classMd.addObject()
                pairRow = emlib.metadata.Row()
                pairRow.setValue(emlib.MDL_IMAGE, getImageLocation(uImg))
                uCoord = uImg.getCoordinate()
                micId = uCoord.getMicId()
                uMic = uMics[micId]
                angles = sangles[micId]
                pairRow.setValue(emlib.MDL_MICROGRAPH, uMic.getFileName())
                pairRow.setValue(emlib.MDL_XCOOR, uCoord.getX())
                pairRow.setValue(emlib.MDL_YCOOR, uCoord.getY())
                pairRow.setValue(emlib.MDL_ENABLED, 1)
                pairRow.setValue(emlib.MDL_ITEM_ID, int(imgId))
                pairRow.setValue(emlib.MDL_REF, 1)
    
                alignment = img.getTransform()
    
                # Scale alignment by scaleFactor
                alignment.scale(scaleFactor)
                alignmentToRow(alignment, pairRow, alignType=ALIGN_2D)
                                   
                pairRow.setValue(emlib.MDL_IMAGE_TILTED, getImageLocation(tImg))
                tMic = tMics[micId]
                pairRow.setValue(emlib.MDL_MICROGRAPH_TILTED, tMic.getFileName())
                (angleY, angleY2, angleTilt) = angles.getAngles()
                pairRow.setValue(emlib.MDL_ANGLE_Y, float(angleY))
                pairRow.setValue(emlib.MDL_ANGLE_Y2, float(angleY2))
                pairRow.setValue(emlib.MDL_ANGLE_TILT, float(angleTilt))
                
                pairRow.writeToMd(classMd, objId)
        
        classMd.write(blockMd, emlib.MD_APPEND)
            
    def _reconstructImages(self, particles, deps):
        """ Function to insert the step needed to reconstruct a
        class (or setOfParticles) """
        classNo = particles.getObjId()
        blockMd = "class%06d_images@%s" % (classNo, self.rctClassesFn)
        classNameIn = blockMd
        classNameOut = self._getExtraPath("rct_images_%06d.xmd" % classNo)
        classVolumeOut = self._getPath("rct_%06d.vol" % classNo)
        
        if particles.hasRepresentative():
            classImage = getImageLocation(particles.getRepresentative())
        else:
            classImage = None
        
        reconStep = self._insertFunctionStep('reconstructClass',
                                             classNameIn, classNameOut,
                                             classImage, classVolumeOut,
                                             prerequisites=[deps])
        return reconStep

    def reconstructClass(self, classIn, classOut, classImage, classVolumeOut):
        # If class image doesn't exists, generate it by averaging
        if classImage is None:
            classRootOut = classOut.replace(".xmd", "") + "_"
            statsFn = self._getExtraPath('stats.xmd')
            args = "-i %(classIn)s --save_image_stats %(classRootOut)s -o %(statsFn)s"
            self.runJob("xmipp_image_statistics", args % locals(),
                        numberOfMpi=1, numberOfThreads=1)

            classImage = classRootOut + "average.xmp"
            
        centerMaxShift = self.maxShift.get()

        args = "-i %(classIn)s -o %(classOut)s --ref %(classImage)s " % locals()
        args += " --max_shift %(centerMaxShift)d" % locals()
        
        if self.thinObject.get():
            args += " --do_stretch"
        
        if self.skipTranslation.get():
            args += " --do_not_align_tilted"
        
        self.runJob("xmipp_image_align_tilt_pairs", args,
                    numberOfMpi=1, numberOfThreads=1)
        
        reconstructAdditionalParams = self.additionalParams.get()

        args = "-i %(classOut)s -o %(classVolumeOut)s " % locals()
        args += " %(reconstructAdditionalParams)s" % locals()

        if self.numberOfMpi >1:
            # threading is not supported in mpi version
            self.runJob("xmipp_reconstruct_art", args, numberOfThreads=1)
        else:
            args += " --thr %d" % self.numberOfThreads.get()
            self.runJob("xmipp_reconstruct_art", args)

        if exists(classVolumeOut):
            mdFn = self._getPath('volumes.xmd')
            md = emlib.MetaData()
            
            if exists(mdFn):
                md.read(mdFn)
            objId = md.addObject()
            md.setValue(emlib.MDL_IMAGE, classVolumeOut, objId)
                        
            if self.doFilter.get():
                filteredVolume = classVolumeOut.replace('.vol', '_filtered.vol')
                lowPassFilter = self.resoLowPassFilter.get()
                args = "-i %(classVolumeOut)s -o %(filteredVolume)s " % locals()
                args += " --fourier low_pass %(lowPassFilter)f " % locals()
                args += " --thr %d" % self.numberOfThreads.get()
                self.runJob("xmipp_transform_filter", args, numberOfMpi=1)
                objId = md.addObject()
                md.setValue(emlib.MDL_IMAGE, filteredVolume, objId)
            md.write(mdFn)
                    
    def createOutputStep(self):
        # TODO: Refactor following code if possible
        self.volumesSet = self._createSetOfVolumes()
        self.volumesSet.setStore(False)
        self.sampling = self.inputParticlesTiltPair.get().getUntilted().getSamplingRate()
        self.volumesSet.setSamplingRate(self.sampling)

        if self.doFilter.get():
            self.volumesFilterSet = self._createSetOfVolumes('filtered')
            self.volumesFilterSet.setStore(False)
            self.volumesFilterSet.setSamplingRate(self.sampling)

        if isinstance(self.inputSet, SetOfParticles):
            volumeOut = self._getPath("rct_%06d.vol" % self.inputSet.getObjId())
            self._appendOutputVolume(volumeOut)
        else:
            for class2D in self.inputSet:
                volumeOut = self._getPath("rct_%06d.vol" % class2D.getObjId())
                self._appendOutputVolume(volumeOut)

        self._defineOutputs(outputVolumes=self.volumesSet)

        if self.doFilter.get():
            self._defineOutputs(outputFilteredVolumes=self.volumesFilterSet)

        self._defineSourceRelation(self.inputParticlesTiltPair, self.volumesSet)

    def _appendOutputVolume(self, volumeOut):
        fnMrc = convertToMrc(self, volumeOut, self.sampling, True)
        vol = Volume()
        vol.setFileName(fnMrc)
        vol.setSamplingRate(self.sampling)
        self.volumesSet.append(vol)

        if self.doFilter.get():
            volumeFilterOut = volumeOut.replace('.vol', '_filtered.vol')
            fnMrc = convertToMrc(self, volumeFilterOut, self.sampling, True)
            volf = Volume()
            volf.setFileName(fnMrc)
            volf.setSamplingRate(self.sampling)
            self.volumesFilterSet.append(volf)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []

        if isinstance(self.inputParticles.get(), SetOfParticles):
            if not self.inputParticles.get().hasAlignment():
                errors.append("The input particles should have "
                              "alignment information.")
        else:
            for class2D in self.inputParticles.get():
                if not class2D.hasAlignment():
                    errors.append("The input classes should have "
                                  "alignment information.")

        return errors
    
    def _summary(self):
        summary = []

        if not hasattr(self, 'outputVolumes'):
            summary.append("Output volumes not ready yet.")
        else:
            if isinstance(self.inputParticles.get(), SetOfParticles):
                summary.append("Input particles: %d"
                               % self.inputParticles.get().getSize())
            else:
                summary.append("Input classes: %d"
                               % self.inputParticles.get().getSize())
            summary.append("Output volumes: %d"
                           % self.outputVolumes.getSize())

            if self.doFilter.get():
                summary.append("Output filtered volumes: %d"
                               % self.outputFilteredVolumes.getSize())

        return summary
        
    def _methods(self):
        methods = []

        if not hasattr(self, 'outputVolumes'):
            methods.append("Output volumes not ready yet.")
        else:
            if isinstance(self.inputParticles.get(), SetOfParticles):
                methods.append('Set of %d particles %s was employed to create '
                               'an initial volume using RCT method.'
                               % (len(self.inputParticles.get()),
                                  self.getObjectTag('inputParticles')))
            else:
                particlesArray = [len(s) for s in self.inputParticles.get()]
                particlesArrayString = String(particlesArray)
                methods.append('Set of %d classes %s was employed to create %d '
                               'initial volumes using RCT method. '
                               % (len(self.inputParticles.get()),
                                  self.getObjectTag('inputParticles'),
                                  len(self.inputParticles.get())))
                methods.append('For each initial volume were used respectively '
                               '%s particles' % particlesArrayString)

            methods.append("Output volumes: %s" %
                           self.getObjectTag('outputVolumes'))

            if self.doFilter.get():
                methods.append("Output filtered volumes: %s"
                               % self.getObjectTag('outputFilteredVolumes'))

        return methods
            
    def _citations(self):
        return ['Sorzano2015b']
