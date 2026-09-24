# **************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
# *              Javier Mota (jmota@cnb.csic.es)
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

import os

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, BooleanParam
from pyworkflow.utils.path import cleanPath
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Image
import pwem.emlib.metadata as md

from pwem import emlib
from xmipp3.convert import setXmippAttributes, xmippToLocation

        
class XmippProtGenerateReprojections(ProtAnalysis3D):
    """Compares a set of classes or averages with the corresponding projections of a reference volume.
    The set of images must have a 3D angular assignment and the protocol computes the residues
    (the difference between the experimental images and the reprojections). The zscore of the mean
    and variance of the residues are computed. Large values of these scores may indicate outliers.
    The protocol also analyze the covariance matrix of the residual and computes the logarithm of
    its determinant [Cherian2013]. The extremes of this score (called zScoreResCov), that is
    values particularly low or high, may indicate outliers.

    AI Generated

    ## Overview

    The Generate Reprojections protocol compares a set of experimental particle
    images with the corresponding projections of a reference 3D volume.

    The input images must already have a 3D angular assignment. This means that
    each image is associated with an orientation describing from which direction it
    is expected to view the 3D structure. Using these orientations, the protocol
    projects the input volume and generates synthetic reference images that can be
    directly compared with the experimental ones.

    This protocol is useful for visual inspection, validation, outlier detection,
    and quality assessment. It helps the user answer a simple but important
    question: given the current angular assignment and the reference volume, do the
    experimental images look like the projections expected from the model?

    The protocol produces two related outputs: one output set containing the
    experimental images with updated Xmipp metadata, and another output set
    containing the generated reprojections. These two outputs can be compared to
    evaluate agreement between data and model.

    ## Inputs and General Workflow

    The protocol requires two main inputs:

    - a set of input particles or images;
    - a reference volume.

    The input image set must contain projection-alignment information. In practical
    terms, this means that the particles, classes, or averages must already have
    assigned Euler angles and shifts from a previous 3D assignment, refinement, or
    classification step.

    The reference volume is converted to Xmipp format and, if necessary, resized so
    that its dimensions match the dimensions of the input images. The protocol then
    uses the angular information associated with each image to generate the
    corresponding projection of the volume.

    For each input image, the protocol stores the relation between the experimental
    image and the generated reference projection. This makes it possible to inspect
    the two sets side by side or to use them in later analysis.

    ## Input Images

    The **Input images** parameter should point to a set of particles or image
    averages with valid 3D angular assignments.

    This is essential. The protocol does not perform an initial orientation search
    from scratch in the usual sense of a full refinement protocol. Instead, it uses
    the angular information already associated with the images to generate the
    corresponding reprojections of the volume.

    Typical inputs may include:

    - particles after a 3D refinement;
    - class averages with assigned projection directions;
    - particles or averages produced by a previous Xmipp angular assignment step.

    If the input images do not have meaningful angular assignments, the generated
    reprojections will not be biologically interpretable. The comparison is only as
    good as the orientations provided in the input set.

    ## Reference Volume

    The **Volume to compare images to** is the 3D map that will be projected.

    The volume should represent the structure that the input images are expected to
    show. In many workflows, this will be the current refined map, an initial model,
    or a representative volume from a 3D classification result.

    Before generating reprojections, the protocol checks the size of the volume
    relative to the input images. If the volume dimension differs from the image
    dimension, the volume is resized to match the image size. This ensures that the
    generated projections can be directly compared with the input images.

    From a biological point of view, the reference volume defines the structural
    hypothesis being tested. If the volume does not correspond to the particles, or
    if it represents a different conformation, the reprojections may differ
    substantially from the experimental images.

    ## Ignore CTF

    The **Ignore CTF** option controls whether the contrast transfer function is
    considered when generating reprojections.

    If **Ignore CTF** is enabled, the protocol generates projections that look more
    like ideal projections of the 3D volume. This is often easier for a human user
    to interpret visually because the projections resemble the structural content
    of the map without microscope-induced CTF modulation.

    If **Ignore CTF** is disabled, the generated projections include the effect of
    the CTF when such information is available. These projections are closer to
    what the microscope would actually record for each particle. This can be more
    appropriate for quantitative comparison, but the images may be less intuitive
    to inspect visually because CTF oscillations and contrast inversions can affect
    their appearance.

    For visual comparison and user interpretation, ignoring the CTF is often a good
    starting point. For more microscope-realistic comparisons, the CTF should be
    included.

    ## Reprojections and Experimental Images

    For each input image, the protocol generates the projection of the reference
    volume corresponding to the image orientation.

    The experimental image and the generated reprojection should be interpreted as
    a pair:

    - the experimental image is what was observed in the data;
    - the reprojection is what the reference volume predicts from the same viewing
      direction.

    Good agreement suggests that the angular assignment and the reference volume
    are consistent with the experimental image. Poor agreement may indicate a
    wrong orientation, a bad particle, conformational heterogeneity, incorrect CTF
    handling, or a mismatch between the particle and the reference map.

    This comparison is particularly useful when inspecting suspicious particles,
    classes, or views that appear inconsistent with the current 3D model.

    ## Output Particles

    The first output, **outputParticles**, contains the experimental images selected
    from the input set, enriched with Xmipp metadata linking them to the generated
    reference projections.

    These images keep the relevant information from the original input particles,
    including sampling and alignment information. The protocol also stores
    metadata fields that point to the original image and to the corresponding
    reference projection.

    This output is useful when the user wants to continue working with the
    experimental images while preserving the connection to their model-based
    reprojections.

    ## Output Projections

    The second output, **outputProjections**, contains the generated projections of
    the reference volume.

    Each image in this set corresponds to one input image and is generated using
    the angular assignment associated with that input image. Therefore, the order
    and correspondence between the experimental images and the reprojections should
    be preserved.

    This output is useful for visual inspection and for direct comparison with the
    input particles or averages. For example, the user can compare experimental
    images and reprojections to assess whether the reference volume explains the
    observed views.

    ## Interpreting Differences Between Images and Reprojections

    Differences between experimental images and reprojections can have several
    causes.

    Some differences are expected because experimental cryo-EM particles are noisy,
    affected by CTF, and may contain small alignment errors. A reprojection of a
    clean 3D map will usually look cleaner and more regular than a raw particle.

    Large or systematic differences may be more informative. They can suggest that
    some particles are incorrectly aligned, that they belong to another
    conformational state, that the reference volume is incomplete or biased, or
    that some particles are contaminants or damaged views.

    When the input images are class averages, the comparison is often easier to
    interpret because averaging reduces noise. In that case, clear differences
    between class averages and reprojections may point to genuine structural
    heterogeneity or model mismatch.

    ## Practical Recommendations

    Use this protocol after a 3D refinement, angular assignment, or classification
    step that has produced meaningful projection directions.

    Start with **Ignore CTF** enabled if the goal is visual interpretation. This
    usually produces cleaner and more intuitive reprojections of the reference map.

    Disable **Ignore CTF** when the goal is to compare the images with projections
    that are closer to the microscope-recorded signal, provided that reliable CTF
    information is available.

    Always check that the input image set and the reference volume correspond to
    the same particle type, box size, and approximate structural state. The
    protocol can resize the volume to match the image dimensions, but it cannot
    correct a biologically inappropriate reference.

    When possible, inspect experimental images and reprojections side by side.
    Look for agreement in the main structural features, not for pixel-perfect
    identity. Cryo-EM images are noisy and affected by many acquisition and
    processing factors.

    This protocol is especially useful for identifying outlier classes, suspicious
    orientations, or particles that do not agree with the current 3D model.

    ## Final Perspective

    Generate Reprojections is a model-comparison protocol. It takes the current
    3D structural hypothesis and asks how the input images should look if that
    hypothesis and the assigned orientations are correct.

    For biological users, the value of the protocol lies in making the relationship
    between particles and volume visible. It helps assess whether the 3D map
    explains the experimental data, whether some views are problematic, and whether
    the current angular assignment is plausible.

    Used carefully, this protocol provides an intuitive bridge between 2D particle
    images and the 3D volume reconstructed from them.
    """
    _label = 'generate reprojections'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume to compare images to", important=True,
                      pointerClass='Volume',
                      help='Volume to be used for class comparison')
        form.addParam('ignoreCTF', BooleanParam, default=True, label='Ignore CTF',
                      help='By ignoring the CTF you will create projections more similar to what a person expects, '
                           'while by using the CTF you will create projections more similar to what the microscope sees')
        form.addParallelSection(threads=0, mpi=8)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        vol = self.inputVolume.get()
        
        self._insertFunctionStep("convertStep")
        imgSet = self.inputSet.get()
        anglesFn = self.imgsFn
        self._insertFunctionStep("produceProjections",
                                 anglesFn,
                                 vol.getSamplingRate())
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from xmipp3.convert import writeSetOfParticles
        imgSet = self.inputSet.get()
        writeSetOfParticles(imgSet, self.imgsFn)

        from pwem.emlib.image import ImageHandler
        img = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xdim = self.inputVolume.get().getDim()[0]

        imgXdim = imgSet.getDim()[0]
        if xdim != imgXdim:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" % (fnVol, imgXdim), numberOfMpi=1)
    
    def produceProjections(self, fnAngles, Ts):
        fnVol = self._getTmpPath("volume.vol")
        anglesOutFn = self._getExtraPath("anglesCont.stk")
        projectionsOutFn = self._getExtraPath("projections.stk")
        args = "-i %s -o %s --ref %s --oprojections %s --sampling %f " %\
             (fnAngles, anglesOutFn, fnVol, projectionsOutFn, Ts)
        if self.ignoreCTF:
            args += " --ignoreCTF "
        self.runJob("xmipp_angular_continuous_assign2", args)
        fnNewParticles = self._getExtraPath("images.stk")
        if os.path.exists(fnNewParticles):
            cleanPath(fnNewParticles)
    
    def createOutputStep(self):

        fnImgs = self._getExtraPath('images.stk')
        if os.path.exists(fnImgs):
            cleanPath(fnImgs)

        imgSet = self.inputSet.get()
        imgFn = self._getExtraPath("anglesCont.xmd")

        self.newAssignmentPerformed = os.path.exists(self._getExtraPath("angles.xmd"))
        self.samplingRate = self.inputSet.get().getSamplingRate()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        if not self.newAssignmentPerformed:
            outputSet.setAlignmentProj()
        self.iterMd = md.iterRows(imgFn, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(imgSet,
                            updateItemCallback=self._updateItem
                            )
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

        imgSet = self.inputSet.get()
        outputSet2 = self._createSetOfParticles('2')
        outputSet2.copyInfo(imgSet)
        if not self.newAssignmentPerformed:
            outputSet2.setAlignmentProj()
        self.iterMd2 = md.iterRows(imgFn, md.MDL_ITEM_ID)
        self.lastRow2 = next(self.iterMd2)
        outputSet2.copyItems(imgSet,
                            updateItemCallback=self._updateItem2,
                            )
        self._defineOutputs(outputProjections=outputSet2)
        self._defineSourceRelation(self.inputSet, outputSet2)

    def _updateItem(self, particle, row):
        count = 0

        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                md.MDL_ITEM_ID):
            count += 1
            if count:
                self._processRow(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0

    def _updateItem2(self, particle2, row):
        count = 0

        while self.lastRow2 and particle2.getObjId() == self.lastRow2.getValue(
                md.MDL_ITEM_ID):
            count += 1
            if count:
                self._processRow2(particle2, self.lastRow2)
            try:
                self.lastRow2 = next(self.iterMd2)
            except StopIteration:
                self.lastRow2 = None

        particle2._appendItem = count > 0

    def _processRow(self, particle, row):
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))

        particle.setLocation(xmippToLocation(row.getValue(emlib.MDL_IMAGE)))
        #__setXmippImage(emlib.MDL_IMAGE)
        #__setXmippImage(emlib.MDL_IMAGE_REF)

        setXmippAttributes(particle, row, emlib.MDL_IMAGE_ORIGINAL,
                           emlib.MDL_IMAGE_REF)

    def _processRow2(self, particle, row):
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))

        particle.setLocation(xmippToLocation(row.getValue(
            emlib.MDL_IMAGE_REF)))
        #__setXmippImage(emlib.MDL_IMAGE)
        #__setXmippImage(emlib.MDL_IMAGE_REF)

        setXmippAttributes(particle, row, emlib.MDL_IMAGE_ORIGINAL,
                           emlib.MDL_IMAGE_REF)


    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append(
                "We projected the volume %s following the directions in %i input images %s." \
                % (
                self.getObjectTag('inputVolume'), self.inputSet.get().getSize(),
                self.getObjectTag('inputSet')))
        return methods

