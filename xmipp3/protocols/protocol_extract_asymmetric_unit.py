# **************************************************************************
# * Authors:     Marta Martinez (mmmtnez@cnb.csic.es)
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

from pyworkflow import VERSION_3_0
from pwem.objects import Volume
from pwem.constants import (SYM_DIHEDRAL_X,SCIPION_SYM_NAME)
from pwem.objects import Transform
from pwem.convert import Ccp4Header
from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        EnumParam, IntParam, GE)

from xmipp3.constants import (XMIPP_SYM_NAME, XMIPP_TO_SCIPION, XMIPP_CYCLIC,
                              XMIPP_DIHEDRAL_X, XMIPP_TETRAHEDRAL, XMIPP_OCTAHEDRAL,
                              XMIPP_I222, XMIPP_I222r, XMIPP_In25, XMIPP_In25r)

DEBUG = True


class XmippProtExtractUnit(EMProtocol):
    """ Generates the necessary files for volumes and Fourier Shell Correlation
    (FSC) curves to submit structural data to the Electron Microscopy Data Bank
    (EMDB). This protocol ensures proper formatting and data preparation for
    public deposition.

    AI Generated

    ## Overview

    The Extract Asymmetric Unit protocol crops from a symmetric 3D map the
    region corresponding to a single asymmetric unit, or more generally to the
    unit cell defined by the selected symmetry. Its main purpose is to isolate
    the smallest unique structural region that, by application of the symmetry
    operators, can reproduce the full volume.

    In practical cryo-EM workflows, this protocol is useful when one wants to
    inspect, analyze, or process only one symmetry-related portion of a map
    instead of the full reconstruction. This is often relevant for symmetric
    macromolecular assemblies, where the biologically meaningful interpretation
    may require focusing on one subunit, one protomer, or one repeated region
    without the redundancy of the complete symmetric map.

    For a biological user, this protocol is essentially a symmetry-aware
    cropping tool. It does not refine the map or change its resolution; rather,
    it extracts the map region corresponding to one symmetry unit in a
    geometrically consistent way.

    ## Inputs and General Workflow

    The protocol requires an **input volume** and a description of its
    **symmetry**. Based on that symmetry, Xmipp identifies the region of space
    corresponding to one asymmetric unit and extracts it from the original map.

    The extraction is additionally controlled by radial limits and an optional
    expansion factor. These parameters define how tightly or loosely the
    cropped region should follow the nominal asymmetric unit.

    The output is a new 3D volume containing only the selected unit, with its
    sampling rate and origin preserved consistently.

    ## Understanding the Asymmetric Unit

    In a symmetric reconstruction, many regions of the map are equivalent by
    symmetry. The **asymmetric unit** is the smallest non-redundant portion
    needed to generate the whole structure by applying the symmetry operations.

    Biologically, this is often the most natural region for focused
    interpretation. For example, in a cyclic or dihedral assembly, one may want
    to inspect a single repeating subunit; in an icosahedral map, one may want
    to isolate one symmetry-related wedge of density.

    It is important to note that the extracted asymmetric unit is defined
    geometrically by the symmetry operators, not necessarily by biochemical
    boundaries. In some cases, a biological subunit may cross the boundaries of
    the formal asymmetric unit. Therefore, the result should always be
    interpreted with the biology of the complex in mind.

    ## Choosing the Symmetry

    The most important parameter is the **symmetry group**. The protocol
    supports cyclic, dihedral, tetrahedral, octahedral, and several icosahedral
    conventions.

    This choice must match the symmetry actually imposed or assumed in the
    reconstruction. If the wrong symmetry is selected, the extracted region will
    not correspond to the intended structural unit and may be difficult or
    impossible to interpret biologically.

    For **cyclic** and **dihedral** symmetry, the **symmetry order** must also
    be provided. This defines, for example, whether the map is C3, C6, D2, D7,
    and so on.

    In practice, users should use exactly the same symmetry convention that was
    used in the reconstruction or in the downstream interpretation of the map.

    ## Symmetry Offset

    For cyclic and dihedral symmetries, the protocol allows an **offset** in
    degrees. This rotates the extracted unit around the symmetry axis before
    cropping.

    This parameter is useful when the default asymmetric unit boundary is
    geometrically correct but not the most convenient one for interpretation.
    By rotating the unit cell, the user can choose a different
    symmetry-equivalent wedge of the volume.

    From a biological point of view, this can be valuable when one wants the
    extracted region to better match a recognizable subunit or to avoid
    cutting through an especially important density region.

    ## Inner and Outer Radius

    The extracted region is also constrained by an **inner radius** and an
    **outer radius**, both expressed in pixels.

    The **inner radius** defines an internal exclusion zone. This is useful
    when the central part of the map is not relevant to the unit one wants to
    isolate, or when one wants to avoid including density near the symmetry axis.

    The **outer radius** defines the maximum radial extent of the cropped
    region. If this parameter is set to -1, the protocol automatically uses
    half the box size of the volume.

    In practical terms, these radii allow the user to restrict the asymmetric
    unit to the radial shell where the relevant density lies. This can be
    particularly helpful in hollow assemblies, shells, or capsids, where the
    density is concentrated away from the center.

    ## Expand Factor

    The **expand factor** enlarges the extracted region beyond the exact formal
    limits of the asymmetric unit.

    This is often biologically useful because strict symmetry boundaries may
    cut too close to real density, especially when a subunit extends near the
    edge of the formal unit. A small expansion can make the result easier to
    inspect and less prone to truncating relevant structural features.

    However, if the expansion is too large, the extracted region may start to
    include density belonging to neighboring symmetry-related units, which
    reduces the conceptual clarity of isolating one unit.

    A moderate expansion is often a good compromise when the goal is
    visualization or local interpretation.

    ## Origin and Sampling Considerations

    The protocol takes into account the **sampling rate** and the **origin** of
    the input volume when performing the extraction. This is important because
    the asymmetric unit must be cropped in the correct geometric frame.

    The output volume retains the appropriate sampling and is assigned an origin
    derived from the cropped map header. This helps ensure that the result
    remains spatially meaningful and can be used consistently in further
    visualization or analysis.

    For most biological users, this means that the output is not just a cropped
    box, but a properly defined volume with geometry preserved.

    ## Outputs and Their Interpretation

    The protocol produces a single **output volume** containing the extracted
    asymmetric unit.

    This output can be used for:
    * focused visualization of one symmetry-related region,
    * local interpretation of a repeating unit,
    * preparation of figures,
    * comparison of a single unit across related maps,
    * or as an intermediate step before more specialized local analysis.

    Biologically, the extracted map should be interpreted as the density
    corresponding to one symmetry-defined region of the original reconstruction.
    It is not a new reconstruction, and it does not change the information
    content of the original map; it simply isolates one part of it.

    ## Practical Recommendations

    A good first step is to make sure that the symmetry assignment matches the
    original reconstruction exactly. Errors in symmetry choice are the most
    common source of unintuitive results.

    For many datasets, using the default outer radius and an inner radius of
    zero is a reasonable starting point. The radii can then be adjusted if the
    extracted region includes too much empty space or cuts through relevant
    density.

    The expand factor is often useful when the strictly defined asymmetric unit
    looks too tight. A small positive value can improve interpretability,
    especially in protein assemblies where the formal symmetry boundaries do
    not align perfectly with intuitive biochemical subunits.

    When using cyclic or dihedral symmetry, the offset can be explored if the
    default extracted wedge is not the most convenient for visualization.

    ## Final Perspective

    The Extract Asymmetric Unit protocol is a symmetry-aware map extraction tool
    that helps isolate one unique region from a symmetric 3D reconstruction.
    Its main value is interpretative: it allows the user to examine a
    non-redundant unit of the map without the visual and structural redundancy
    of the full symmetric assembly.

    For most cryo-EM users, it is best seen as a focused visualization and
    analysis aid. When the symmetry is correctly defined and the radial limits
    are sensibly chosen, it provides a very useful way to study one part of a
    symmetric map in a cleaner and more biologically manageable form.
    """
    _label = 'extract asymmetric unit'
    _program = ""
    _version = VERSION_3_0

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolumes', PointerParam, label="Input Volume",
                      important=True, pointerClass='Volume',
                      help='This volume will be cropped')
        form.addParam('symmetryGroup', EnumParam,
                      choices=[XMIPP_SYM_NAME[XMIPP_CYCLIC] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_CYCLIC]] + ")",
                               XMIPP_SYM_NAME[XMIPP_DIHEDRAL_X] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_DIHEDRAL_X]] + ")",
                               XMIPP_SYM_NAME[XMIPP_TETRAHEDRAL] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_TETRAHEDRAL]] + ")",
                               XMIPP_SYM_NAME[XMIPP_OCTAHEDRAL] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_OCTAHEDRAL]] + ")",
                               XMIPP_SYM_NAME[XMIPP_I222] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_I222]] + ")",
                               XMIPP_SYM_NAME[XMIPP_I222r] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_I222r]] + ")",
                               XMIPP_SYM_NAME[XMIPP_In25] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_In25]] + ")",
                               XMIPP_SYM_NAME[XMIPP_In25r] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_In25r]] + ")"],
                      default=XMIPP_I222,
                      label="Symmetry",
                      help="See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry"
                           "Symmetry for a description of the symmetry groups "
                           "format in Xmipp.\n"
                           "If no symmetry is present, use _c1_."
                      )
        form.addParam('symmetryOrder', IntParam, default=1,
                      condition='symmetryGroup<=%d' % SYM_DIHEDRAL_X,
                      label='Symmetry Order',
                      help='Order of cyclic symmetry.')
        form.addParam('offset', FloatParam, default=0.,
                      condition='symmetryGroup<=%d' % SYM_DIHEDRAL_X,
                      label="offset",
                      help="rotate unit cell around z-axis by offset degrees")
        form.addParam('innerRadius', FloatParam, default=0.0,
                      label="Inner Radius (px)", validators=[GE(0.0)],
                      help="inner Mask radius")
        form.addParam('outerRadius', FloatParam, default=-1,
                      label="Outer Radius (px)",
                      help="outer Mask radius, if -1, the radius will be "
                           "volume_size/2")
        form.addParam('expandFactor', FloatParam, default=0.,
                      label="Expand Factor",
                      help="Increment cropped region by this factor")

    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.extractUnit)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------

    def extractUnit(self):
        sym = self.symmetryGroup.get()
        if sym == XMIPP_CYCLIC:
            sym = "%s%d" % (XMIPP_SYM_NAME[XMIPP_CYCLIC][:1], self.symmetryOrder)
        elif sym == XMIPP_DIHEDRAL_X:
            sym = "%s%d" % \
                  (XMIPP_SYM_NAME[XMIPP_DIHEDRAL_X][:1], self.symmetryOrder)
        elif sym == XMIPP_TETRAHEDRAL:
            sym = "%s" % XMIPP_SYM_NAME[XMIPP_TETRAHEDRAL]
        elif sym == XMIPP_OCTAHEDRAL:
            sym = "%s" % XMIPP_SYM_NAME[XMIPP_OCTAHEDRAL]
        elif sym >= XMIPP_I222 and sym <= XMIPP_In25r :
            sym = XMIPP_SYM_NAME[sym]
        
        inFileName = self.inputVolumes.get().getFileName()
        if inFileName.endswith('.mrc'):
            inFileName = inFileName + ":mrc"
        args = "-i %s -o %s" % \
               (inFileName, self._getOutputVol())
        args += " --unitcell %s " % sym
        args += " %f " % self._getInnerRadius()
        args += " %f " % self._getOuterRadius()
        args += " %f " % self.expandFactor.get()
        args += " %f " % self.offset.get()
        sampling = self.inputVolumes.get().getSamplingRate()
        args += " %f " % sampling
        origin = self.inputVolumes.get().getShiftsFromOrigin()
        # x origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[0] / (-1. * sampling))
        # y origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[1] / (-1. * sampling))
        # z origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[2] / (-1. * sampling))

        self.runJob("xmipp_transform_window", args)

    def createOutputStep(self):
        vol = Volume()
        vol.setLocation(self._getOutputVol())
        sampling = self.inputVolumes.get().getSamplingRate()
        vol.setSamplingRate(sampling)
        #

        ccp4header = Ccp4Header(self._getOutputVol(), readHeader=True)
        t = Transform()

        x, y, z = ccp4header.getOrigin()  # origin output vol
        # coordinates

        t.setShifts(x, y, z)
        vol.setOrigin(t)
        #
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputVolumes, self.outputVolume)

    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        message = []
        return message

    def _summary(self):
        # message = "Data Available at : *%s*"% self.filesPath.get()
        message = ""
        return [message]

    def _methods(self):
        return []

    # --------------------------- UTILS functions -----------------------------
    def _getInnerRadius(self):
        return self.innerRadius.get()
    
    def _getOuterRadius(self):
        outerRadius = self.outerRadius.get()
        if outerRadius < 0:
            volume: Volume = self.inputVolumes.get()
            dim = volume.getDimensions()
            outerRadius = dim[0] / 2
        return outerRadius
    
    def _getOutputVol(self):
        prefix = os.path.basename(self.inputVolumes.get().getFileName()).split(".")[0]

        return self._getExtraPath(prefix + "_output_volume.mrc")

    def replace_at_index(self, tup, ix, val):
        return tup[:ix] + (val,) + tup[ix + 1:]
