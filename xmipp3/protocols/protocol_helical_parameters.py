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

import pyworkflow.object as pwobj
from pwem.objects import Volume
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtPreprocessVolumes
import pyworkflow.protocol.params as params
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem.emlib import MetaData, MDL_ANGLE_ROT, MDL_SHIFT_Z
from xmipp3.base import HelicalFinder
from xmipp3.convert import getImageLocation


class XmippProtHelicalParameters(ProtPreprocessVolumes, HelicalFinder):
    """Estimates the helical symmetry and parameters of a structure. Helical
    symmetry is defined mathematically as V(r,rot,z)=V(r,rot+k*DeltaRot,z+k*Deltaz)
    and applied to improve the reconstruction of the volume specimens inthe processing.
    You can limit the radios of the helix, apply dihedral symmetry and apply Cn
    symmetry

    AI Generated

    ## Overview

    The Helical Symmetry protocol estimates the helical parameters of a 3D volume
    and uses them to produce a symmetrized version of the structure.

    Helical symmetry describes structures that repeat by a combination of rotation
    and translation along an axis. In this protocol, the helix is assumed to be
    oriented along the Z axis of the input volume. The repeated unit is described
    by two main parameters:

    - **DeltaRot**, the rotation angle between consecutive helical repeats;
    - **DeltaZ**, the axial displacement, or rise, between consecutive repeats.

    In mathematical terms, the density is expected to satisfy a relation of the
    form:

    \[
    V(r,\theta,z) \approx V(r,\theta + k \Delta\theta, z + k \Delta z)
    \]

    for integer values of \(k\), where \(\Delta\theta\) is the helical twist and
    \(\Delta z\) is the helical rise.

    The protocol first searches for the helical parameters that best explain the
    input volume. It then applies the estimated symmetry to generate a symmetrized
    output volume. This can improve the signal-to-noise ratio and produce a more
    consistent reconstruction when the sample truly follows helical symmetry.

    ## Inputs and General Workflow

    The main input is a 3D volume representing a helical specimen. The volume
    should already be approximately aligned so that the helical axis corresponds to
    the Z axis.

    The protocol proceeds in several conceptual steps.

    First, it prepares the input volume. If requested, it may apply an initial
    dihedral symmetry operation before searching for the helical parameters.

    Second, it performs a coarse search over the user-defined ranges of rotation
    and axial shift. This step explores the possible helical twists and rises.

    Third, it performs a fine search around the best candidates found in the coarse
    search. This refines the estimated helical parameters.

    Finally, the protocol symmetrizes the volume using the refined helical
    parameters and produces an output volume. It also reports the estimated
    DeltaRot and DeltaZ values.

    ## Input Volume

    The **Input volume** should be a 3D reconstruction of a helical structure.

    For the search to be meaningful, the helical axis should be approximately
    aligned with the Z axis. If the helix is tilted or strongly off-center, the
    estimated parameters may be unreliable.

    The input volume should also contain enough visible helical signal. Very noisy
    volumes, volumes with strong distortions, or volumes dominated by non-helical
    features may lead to incorrect estimates.

    This protocol is most appropriate when the user already has a preliminary
    helical reconstruction and wants to estimate or refine its symmetry parameters.

    ## Helical Rise and Twist

    The two most important outputs are **DeltaZ** and **DeltaRot**.

    **DeltaZ** is the axial translation between consecutive helical repeats. It is
    reported in angstroms. In the summary, it may also be expressed in voxels by
    dividing by the sampling rate.

    **DeltaRot** is the rotation angle, in degrees, between consecutive helical
    repeats.

    Together, these two values define the helical operation. Repeated application
    of this operation should map one subunit or repeating density element onto the
    next one along the helix.

    Biologically, these parameters describe the architecture of the filament. They
    are related to the pitch, number of subunits per turn, and repeat organization
    of the assembly.

    ## Cylinder Inner and Outer Radius

    The **Cylinder inner radius** and **Cylinder outer radius** restrict the radial
    region of the volume used for symmetry estimation and symmetrization.

    The helix is assumed to occupy a cylindrical region around the Z axis. By
    specifying inner and outer radii, the user can tell the protocol which part of
    the volume should contribute to the search.

    If both values are left as -1, the whole volume is used.

    Using a cylindrical restriction can be useful when the volume contains central
    noise, peripheral artifacts, solvent density, or regions that should not
    contribute to the symmetry search. It can also help focus the search on the
    part of the map where the helical signal is strongest.

    The radii are given in voxels. Users should choose them according to the size
    and position of the filament in the box.

    ## Height Fraction

    The **Height fraction** controls how much of the volume height is used to
    estimate the helical parameters.

    A value of 1 uses the full height of the volume. Values below 1 use only a
    central fraction of the volume. This is useful because the top and bottom
    planes of a reconstruction are often less reliable, affected by edge effects,
    or less well resolved.

    The default value excludes a small fraction of the extremes while keeping most
    of the helix. This is usually a sensible choice.

    If the full volume is well resolved and free of edge artifacts, a value of 1
    can also be appropriate. If the ends of the helix are noisy or distorted, a
    smaller fraction may improve the robustness of the search.

    ## Search Limits for Rotation

    The parameters **Minimum rotational angle**, **Maximum rotational angle**, and
    **Angular step** define the coarse search range for DeltaRot.

    The protocol explores rotational angles in this interval using the specified
    step size. A smaller angular step gives a finer search but increases
    computation time. A larger step is faster but may miss the best value or give a
    less accurate starting point for the fine search.

    If little is known about the helix, a broad range such as 0 to 360 degrees may
    be used. If approximate helical symmetry is known from prior information,
    literature, indexing, or visual analysis, the range can be narrowed to make the
    search more efficient and less ambiguous.

    ## Search Limits for Axial Shift

    The parameters **Minimum shift Z**, **Maximum shift Z**, and **Shift step**
    define the search range for DeltaZ.

    These values are expressed in angstroms. The minimum axial shift must be
    positive. A zero or negative value is not meaningful for the helical rise and
    is rejected by the protocol.

    As with the rotational search, a smaller step gives a finer exploration but
    requires more computation. A broader range is useful when the rise is unknown,
    whereas a narrower range is preferable when prior structural information is
    available.

    The selected range should contain the expected rise between neighboring
    repeating units. If the true rise lies outside the search interval, the
    protocol cannot recover the correct helical parameters.

    ## Coarse and Fine Search

    The protocol performs two searches.

    The **coarse search** explores the user-defined range of rotations and axial
    shifts. Its purpose is to identify promising helical parameters.

    The **fine search** refines the result from the coarse search. It uses the
    coarse-search output as a starting point and estimates more accurate values for
    DeltaRot and DeltaZ.

    This two-stage strategy is useful because helical symmetry searches can be
    ambiguous. A broad coarse search reduces the risk of missing the correct region,
    whereas a fine search improves the final parameter estimate.

    ## Dihedral Symmetry

    The option **Apply dihedral symmetry** applies dihedral symmetry during the
    preparation and search.

    Dihedral symmetry may be appropriate for some helical assemblies that have an
    additional twofold relationship perpendicular to the helical axis. When this
    symmetry is biologically justified, applying it can improve the signal and make
    the helical search more stable.

    However, dihedral symmetry should not be applied simply because it improves the
    appearance of the map. If the specimen does not truly have this symmetry,
    forcing it may introduce artificial density and distort the biological
    interpretation.

    The advanced option **Force the dihedral axis to be in X** assumes that the
    dihedral axis is around X instead of searching for it. This should only be used
    when the orientation of the volume is known and the user is confident that this
    constraint is correct.

    ## Additional Cn Symmetry

    The option **Apply Cn symmetry** allows the user to impose an additional cyclic
    symmetry around the helical axis.

    For example, C2 symmetry means that the structure is expected to have a
    twofold rotational symmetry around the relevant axis. Other Cn values may be
    specified according to the biological symmetry of the assembly.

    This option should be used only when there is prior evidence for the additional
    cyclic symmetry. Applying an incorrect Cn symmetry can average together
    non-equivalent features and damage the interpretation of the map.

    If no additional cyclic symmetry is required, the protocol uses C1, meaning no
    extra cyclic symmetry.

    ## Symmetrized Output Volume

    The main output is a **symmetrized volume**.

    This volume is produced by applying the estimated helical parameters to the
    input volume. If additional dihedral or Cn symmetry was requested, those
    symmetry operations are also included.

    The output volume keeps the sampling information from the input volume. It can
    be used for visualization, further interpretation, or as an improved reference
    in subsequent processing steps.

    The symmetrized volume should be inspected carefully. Symmetry application can
    increase signal-to-noise ratio and improve the clarity of repeating features,
    but it can also amplify errors if the estimated parameters or imposed
    symmetries are wrong.

    ## Output Helical Parameters

    In addition to the symmetrized volume, the protocol reports:

    - **DeltaRot**, the estimated helical rotation in degrees;
    - **DeltaZ**, the estimated helical rise in angstroms.

    These values summarize the helical symmetry found by the protocol. They are
    important not only as processing parameters, but also as structural descriptors
    of the specimen.

    Users should compare these values with prior biological knowledge, expected
    repeat distances, known filament geometry, or independent estimates when
    available.

    ## Practical Recommendations

    Before running the protocol, make sure that the helix is approximately centered
    and aligned with the Z axis. This is one of the most important practical
    conditions for a reliable result.

    Use broad search ranges when the helical parameters are unknown. Use narrower
    ranges when prior information is available, because this reduces ambiguity and
    computation time.

    Keep the height fraction below 1 if the ends of the volume are noisy or poorly
    resolved. Use 1 only when the full height of the reconstruction is reliable.

    Use cylinder radii to focus the search on the meaningful helical density,
    especially when the box contains strong solvent noise, artifacts, or regions
    that should not contribute to symmetry estimation.

    Apply dihedral or Cn symmetry only when it is biologically justified. Incorrect
    extra symmetry can produce attractive but misleading maps.

    After the protocol finishes, inspect the symmetrized volume and verify that the
    main structural features are improved rather than artificially smeared or
    duplicated. Also check that the estimated DeltaZ and DeltaRot are physically
    reasonable for the specimen.

    ## Final Perspective

    The Helical Symmetry protocol estimates the rise and twist that best describe a
    helical volume and uses those parameters to generate a symmetrized map.

    For biological users, this protocol connects image processing with structural
    interpretation. The estimated helical parameters describe how the molecular
    subunits repeat along the filament, while the symmetrized volume can improve
    the visibility of recurring structural features.

    The protocol is most powerful when used with a well-centered, Z-aligned
    preliminary reconstruction and with search ranges guided by biological or
    structural expectations. Used carefully, it can improve both the quality of the
    map and the understanding of the helical organization of the specimen.
    """
    _label = 'helical symmetry'
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume", label='Input volume')
        form.addParam('cylinderInnerRadius', params.IntParam,label='Cylinder inner radius', default=-1,
                      help="The helix is supposed to occupy this radius in voxels around the Z axis. Leave it as -1 for symmetrizing the whole volume")
        form.addParam('cylinderOuterRadius',params.IntParam,label='Cylinder outer radius', default=-1,
                      help="The helix is supposed to occupy this radius in voxels around the Z axis. Leave it as -1 for symmetrizing the whole volume")
        form.addParam('dihedral',params.BooleanParam,default=False,label='Apply dihedral symmetry')
        form.addParam('forceDihedralX',params.BooleanParam,default=False,expertLevel=LEVEL_ADVANCED, label='Force the dihedral axis to be in X',
                      help="If this option is chosen, then the dihedral axis is not searched and it is assumed that it is around X.")
        form.addParam('additionalCn',params.BooleanParam,default=False,label='Apply Cn symmetry')
        form.addParam('Cn',params.StringParam,default="C2",condition="additionalCn",label='Cn symmetry')

        form.addSection(label='Search limits')
        form.addParam('heightFraction',params.FloatParam,default=0.9,label='Height fraction',
                      help="The helical parameters are only sought using the fraction indicated by this number. "\
                           "In this way, you can avoid including planes that are poorly resolved at the extremes of the volume. " \
                           "However, note that the algorithm can perfectly work with a fraction of 1.")
        form.addParam('rot0',params.FloatParam,default=0,label='Minimum rotational angle',help="In degrees")
        form.addParam('rotF',params.FloatParam,default=360,label='Maximum rotational angle',help="In degrees")
        form.addParam('rotStep',params.FloatParam,default=5,label='Angular step',help="In degrees")
        form.addParam('z0',params.FloatParam,default=1,label='Minimum shift Z',help="In Angstroms")
        form.addParam('zF',params.FloatParam,default=10,label='Maximum shift Z',help="In Angstroms")
        form.addParam('zStep',params.FloatParam,default=0.5,label='Shift step',help="In Angstroms")
        self.deltaZ= params.Float()
        self.deltaRot=params.Float()
        form.addParallelSection(threads=4, mpi=0)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('copyInput')
        self._insertFunctionStep('coarseSearch')
        self._insertFunctionStep('fineSearch')
        self._insertFunctionStep('symmetrize')
        self._insertFunctionStep('createOutput')
        self.fnVol = getImageLocation(self.inputVolume.get())
        self.fnVolSym=self._getPath('volume_symmetrized.mrc')
        [self.height,_,_]=self.inputVolume.get().getDim()

    def _getFileName(self, key, **kwargs):
        if key=="fine":
            return self._getExtraPath('fineParams.xmd')
        elif key=="coarse":
            return self._getExtraPath('coarseParams.xmd')
        else:
            return ""
    
    #--------------------------- STEPS functions --------------------------------------------
    def copyInput(self):
        if self.dihedral:
            if not self.forceDihedralX:
                self.runJob("xmipp_transform_symmetrize","-i %s -o %s --sym dihedral --dont_wrap" % (self.fnVol, self.fnVolSym))
            else:
                self.runJob("xmipp_transform_geometry","-i %s -o %s --rotate_volume axis 180 1 0 0" % (self.fnVol, self.fnVolSym))
                self.runJob("xmipp_image_operate","-i %s --plus %s -o %s" % (self.fnVol, self.fnVolSym, self.fnVolSym))
                self.runJob("xmipp_image_operate","-i %s --mult 0.5" % self.fnVolSym)
        else:
            ImageHandler().convert(self.inputVolume.get(), self.fnVolSym)
                        
    def coarseSearch(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runCoarseSearch(self.fnVolSym,self.dihedral.get(),float(self.heightFraction.get()),
                             float(self.z0.get()),float(self.zF.get()),float(self.zStep.get()),
                             float(self.rot0.get()),float(self.rotF.get()),float(self.rotStep.get()),
                             self.numberOfThreads.get(),self._getFileName('coarse'),
                             int(self.cylinderInnerRadius.get()),int(self.cylinderOuterRadius.get()),int(self.height),
                             self.inputVolume.get().getSamplingRate(), Cn)

    def fineSearch(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runFineSearch(self.fnVolSym, self.dihedral.get(), self._getFileName('coarse'),
                           self._getFileName('fine'),
                           float(self.heightFraction.get()),float(self.z0.get()),float(self.zF.get()),
                           float(self.rot0.get()),float(self.rotF.get()),
                           int(self.cylinderInnerRadius.get()),int(self.cylinderOuterRadius.get()),int(self.height),
                           self.inputVolume.get().getSamplingRate(), Cn)

    def symmetrize(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runSymmetrize(self.fnVolSym, self.dihedral.get(), self._getFileName('fine'), self.fnVolSym,
                           float(self.heightFraction.get()),
                           self.cylinderInnerRadius.get(), self.cylinderOuterRadius.get(), self.height,
                           self.inputVolume.get().getSamplingRate(), Cn)

    def createOutput(self):
        volume = Volume()
        volume.setFileName(self.fnVolSym)
        Ts = self.inputVolume.get().getSamplingRate()
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(self.fnVolSym,Ts))
        volume.copyInfo(self.inputVolume.get())
        self._defineOutputs(outputVolume=volume)
        self._defineTransformRelation(self.inputVolume, self.outputVolume)
        
        md = MetaData(self._getFileName('fine'))
        objId = md.firstObject()
        self._defineOutputs(deltaRot=pwobj.Float(md.getValue(MDL_ANGLE_ROT, objId)),
                            deltaZ=pwobj.Float(md.getValue(MDL_SHIFT_Z, objId)))

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        messages = []
        if self.deltaZ.hasValue():
            messages.append('DeltaZ=%f (voxels) %f (Angstroms)'%(self.deltaZ.get()/self.inputVolume.get().getSamplingRate(),self.deltaZ.get()))
            messages.append('DeltaRot=%f (degrees)'%self.deltaRot.get())      
        return messages

    def _citations(self):
        papers=[]
        return papers

    def _validate(self):
        messages=[]
        if float(self.z0.get())<=0:
            messages.append("z0 should not be negative or zero")
        return messages

    def _methods(self):
        messages = []      
        messages.append('We looked for the helical symmetry parameters of the volume %s using Xmipp [delaRosaTrevin2013].' % self.getObjectTag('inputVolume'))
        if self.deltaZ.hasValue():
            messages.append('We found them to be %f Angstroms and %f degrees.'%(self.deltaZ.get(),self.deltaRot.get()))
            messages.append('We symmetrized %s with these parameters and produced the volume %s.'%(self.getObjectTag('inputVolume'),
                                                                                                  self.getObjectTag('outputVolume')))
            if self.dihedral.get():
                messages.append('We applied dihedral symmetry.')
        return messages

