# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
from pwem.emlib.image import ImageHandler
from pwem.objects import FSC
from pwem.protocols import ProtAnalysis3D
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.constants import STEPS_PARALLEL
import pyworkflow.protocol.params as params
import pwem.emlib.metadata as md

from xmipp3.convert import locationToXmipp


class XmippProtMultipleFSCs(ProtAnalysis3D):
    """
    Compute the FSCs between a reference volume and a set of input volumes.
    A mask can be provided and the volumes are aligned by default.

    AI Generated

    ## Overview

    The Multiple FSCs protocol computes Fourier Shell Correlation curves between
    one reference volume and several input volumes.

    Fourier Shell Correlation, or FSC, is one of the standard tools used in cryo-EM
    to compare 3D maps in Fourier space. It measures how similar two volumes are as
    a function of spatial frequency. High FSC values at low frequencies indicate
    agreement in coarse structural features, while the behavior at higher
    frequencies provides information about the similarity of finer details.

    This protocol is useful when the user wants to compare several volumes against
    the same reference. For example, it can be used to compare different
    reconstructions, different classes, maps obtained with different processing
    parameters, or maps produced by different algorithms.

    The output is a set of FSC curves, one for each input volume compared with the
    reference volume.

    ## Inputs and General Workflow

    The protocol requires:

    - one reference volume;
    - one or more volumes to compare with the reference;
    - optionally, a mask.

    The reference volume is first converted to Xmipp format. If a mask is provided,
    the mask is also converted, resized if necessary, and applied to the reference.

    Each input volume is then converted to Xmipp format and resized if needed so
    that its box size matches the reference. If volume alignment is enabled, the
    input volume is locally aligned to the reference. If a mask is provided, the
    same mask is applied to the input volume.

    Finally, the protocol computes the FSC curve between the reference and each
    input volume and stores the results in a Scipion set of FSC objects.

    ## Reference Volume

    The **Reference volume** is the map against which all other volumes will be
    compared.

    This volume defines the coordinate system, box size, sampling rate, and
    structural reference for the comparison. All FSC curves produced by the
    protocol are computed relative to this reference.

    The reference should therefore be chosen carefully. It may be a trusted
    reconstruction, a consensus map, a previous result, a known reference, or the
    map considered most relevant for the biological question.

    Because all input volumes are compared with the same reference, the resulting
    FSC curves can be interpreted together as a comparative profile of map
    similarity.

    ## Volumes to Compare

    The **Volumes to compare** parameter contains the set of input volumes that
    will each be compared with the reference volume.

    Each input volume produces one FSC curve. The label of each FSC curve is taken
    from the corresponding volume label when available, making it easier to
    identify the curves during inspection.

    The input volumes may come from different processing branches, different
    classification results, different refinement protocols, or different parameter
    settings.

    For meaningful comparison, the volumes should represent the same or related
    structures. If the volumes correspond to different conformations, different
    compositions, or different masks, the FSC curves should be interpreted with
    that biological context in mind.

    ## Volume Size Matching

    Before computing the FSC, each input volume is resized if its box size differs
    from the reference volume.

    This is necessary because FSC comparison requires volumes defined on compatible
    grids. The protocol uses the reference volume dimension as the target size.

    This resizing step makes the technical comparison possible, but users should
    still ensure that the volumes are biologically and geometrically comparable.
    A volume can have the correct box size and still be miscentered, misaligned, or
    represent a different region of the structure.

    ## Align Volumes

    The **Align volumes?** option controls whether each input volume is aligned to
    the reference before FSC computation.

    When this option is enabled, the protocol performs a local volume alignment of
    the input volume against the reference and applies the resulting transformation.
    This is useful when the volumes are already roughly in the same orientation but
    may have small residual shifts or rotations.

    The alignment is local, so the initial orientations should be relatively
    similar. It is not intended to solve a completely unknown orientation
    relationship between unrelated maps.

    If the volumes are already in the same coordinate frame, alignment may still be
    useful to correct small differences. If the user wants to compare maps exactly
    as they are, alignment can be disabled.

    ## Mask

    The **Mask** parameter allows the user to provide a volume mask that is applied
    before computing each FSC.

    Masking restricts the comparison to a specific region of the map. This can be
    important when the user wants to focus on the molecular structure and exclude
    solvent, noise, empty box regions, or irrelevant density.

    If a mask is provided, it is converted to Xmipp format and resized if necessary
    to match the reference volume. The same mask is applied to both the reference
    and each input volume before FSC calculation.

    The mask should be chosen carefully. A mask that is too loose may include too
    much background noise. A mask that is too tight may remove real density or
    introduce edge artifacts that affect the FSC curve.

    For fair comparison across multiple volumes, the same mask is applied to all
    comparisons.

    ## FSC Curves

    Each FSC curve describes the similarity between the reference volume and one
    input volume across spatial frequencies.

    At low spatial frequencies, FSC values usually reflect agreement in overall
    shape and large structural features. At higher spatial frequencies, they
    reflect agreement in finer details.

    A curve that remains high to higher frequencies indicates stronger similarity
    between the two maps at finer levels of detail. A curve that drops early
    indicates that the maps agree only at coarser resolution.

    When comparing several curves, users can assess which input volume is most
    similar to the reference and whether differences occur mainly at low,
    intermediate, or high spatial frequencies.

    ## Interpreting Multiple FSCs

    The main advantage of this protocol is that it produces several FSC curves
    against a common reference.

    This makes it useful for comparative questions such as:

    - Which reconstruction is closest to the reference?
    - Do different processing parameters improve or reduce agreement?
    - Are some classes more similar to the reference than others?
    - Does a mask reveal differences in a specific structural region?
    - Do maps agree globally but differ at high resolution?

    The interpretation should always consider how the input volumes were produced.
    An FSC curve is a measure of map similarity, not a complete biological
    validation by itself.

    ## Outputs and Their Interpretation

    The main output is **outputFSCs**, a set of FSC objects.

    Each FSC object corresponds to one input volume compared with the reference.
    The curve stores spatial frequency values and the corresponding FSC values.

    These FSCs can be plotted and compared within Scipion. The labels of the FSCs
    are derived from the input volume labels when available, which helps identify
    which curve belongs to which map.

    The output does not create new volumes. It only reports similarity curves.

    ## Practical Recommendations

    Use this protocol when several volumes need to be compared against the same
    reference under the same conditions.

    Keep **Align volumes?** enabled when volumes may have small residual
    misalignments but are already roughly in the same orientation.

    Disable alignment if the goal is to evaluate the maps exactly in their current
    coordinate frame.

    Use a mask when the comparison should focus on the molecular region rather
    than solvent or empty box. Use the same mask for all volumes to make the
    comparison fair.

    Inspect the full FSC curves, not only a single resolution number. The shape of
    the curve can reveal whether differences occur at low, intermediate, or high
    spatial frequencies.

    Be cautious when comparing volumes that represent different conformations or
    different compositions. A lower FSC may reflect genuine biological differences,
    not necessarily worse reconstruction quality.

    ## Final Perspective

    Multiple FSCs is a comparative map-validation protocol. It allows the user to
    measure how several volumes relate to a common reference in Fourier space.

    For biological users, the protocol is useful for comparing reconstruction
    branches, evaluating classes, checking alternative processing choices, or
    quantifying map similarity under a common mask and alignment strategy.

    The resulting FSC curves should be interpreted as one component of map
    assessment, together with visual inspection, local resolution, model fit,
    particle behavior, and biological plausibility.
    """
    _label = 'multiple fscs'
    _lastUpdateVersion = VERSION_1_1

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('referenceVolume', params.PointerParam,
                      pointerClass='Volume',
                      label="Reference volume",
                      help='The rest of volumes will be compared to this one')

        form.addParam('inputVolumes', params.MultiPointerParam,
                      pointerClass='Volume',
                      label="Volumes to compare",
                      help='Set of volumes to compare to the reference volume')
        form.addParam('mask', params.PointerParam,
                      pointerClass='VolumeMask', allowsNull=True,
                      label="Mask",
                      help='A mask may be provided and it is applied before '
                           'comparing the different volumes')
        form.addParam('doAlign', params.BooleanParam, default=True,
                      label="Align volumes?",
                      help="Align volumes to reference before comparing. A local "
                           "alignment is performed so the initial orientation "
                           "of the volumes should be relatively similar")
        form.addParallelSection(threads=8, mpi=1)

#--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        stepId = self._insertFunctionStep('prepareReferenceStep',
                                          self.referenceVolume.get().getObjId())
        allVols = []
        for i, vol in enumerate(self.inputVolumes):
            volId = self._insertFunctionStep('compareVolumeStep',
                                             vol.get().getLocation(), i+1,
                                             prerequisites=[stepId])
            allVols.append(volId)

        self._insertFunctionStep('createOutputStep', prerequisites=allVols)

    def _resizeVolume(self, volFn):
        """ Resize input volume if not of the same size of referenceVol """
        refDim = self.referenceVolume.get().getXDim()
        volDim = ImageHandler().getDimensions(volFn)
        if refDim != volDim:
            self.runJob('xmipp_image_resize',
                        "-i %s --dim %d" % (volFn, refDim))

    def _maskVolume(self, volFn):
        """ Mask input volume multiplying by mask.vol. """
        self.runJob("xmipp_image_operate",
                    "-i %s --mult %s" % (volFn, self._getExtraPath("mask.vol")))

    def prepareReferenceStep(self,volId):
        inputMask = self.mask.get()

        ih = ImageHandler()
        fnRef = self._getExtraPath("reference.vol")
        ih.convert(self.referenceVolume.get(), fnRef)

        if inputMask is not None:
            fnMask = self._getExtraPath("mask.vol")
            ih.convert(self.mask.get(), fnMask)
            self._resizeVolume(fnMask)
            self._maskVolume(fnRef)

    def compareVolumeStep(self, volLoc, i):
        fnRef = self._getExtraPath("reference.vol")
        sampling = self.referenceVolume.get().getSamplingRate()
        fnRoot = self._getExtraPath("volume_%02d" % i)
        fnVol = fnRoot + ".vol"
        self.runJob("xmipp_image_convert","-i %s -o %s -t vol"%(locationToXmipp(volLoc[0],volLoc[1]),fnVol))
        
        # Resize if the volume has different size than the reference
        self._resizeVolume(fnVol)

        if self.doAlign: # Align against the reference if selected
            self.runJob('xmipp_volume_align',
                        "--i1 %s --i2 %s --apply --local" % (fnRef, fnVol))

        if self.mask.hasValue(): # Mask volume if input mask
            self._maskVolume(fnVol)

        # Finally compute the FSC
        args = "--ref %s -i %s -o %s_fsc.xmd --sampling_rate %f" % (fnRef, fnVol,
                                                                fnRoot, sampling)
        self.runJob("xmipp_resolution_fsc", args)

    def createOutputStep(self):
        fscSet = self._createSetOfFSCs()

        for i, vol in enumerate(self.inputVolumes):
            index = i + 1
            fnFsc = self._getExtraPath("volume_%02d_fsc.xmd" % index)
            mdFsc = md.MetaData(fnFsc)
            fscLabel = vol.get().getObjLabel() or 'FSC %d' % index
            fsc = FSC(objLabel=fscLabel)
            fsc.loadFromMd(mdFsc, md.MDL_RESOLUTION_FREQ, md.MDL_RESOLUTION_FRC)
            fscSet.append(fsc)

        self._defineOutputs(outputFSCs=fscSet)

        self._defineSourceRelation(self.referenceVolume, fscSet)
        for i, vol in enumerate(self.inputVolumes):
            self._defineSourceRelation(vol, fscSet)

