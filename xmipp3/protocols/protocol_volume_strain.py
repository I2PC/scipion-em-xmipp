# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from pyworkflow.protocol.params import PointerParam, StringParam
from pwem.protocols import ProtAnalysis3D

from xmipp3.base import getXmippPath
from xmipp3 import Plugin


class XmippProtVolumeStrain(ProtAnalysis3D):
    """Compares two volume states to analyze local strains and rotations. This
    protocol helps study structural changes by quantifying deformation and
    dynamic behavior between different conformations.

    AI Generated

    ## Overview

    The Calculate Strain protocol compares two 3D volume states and estimates
    local deformation descriptors between them.

    The protocol is intended for situations where two maps represent different
    states of the same structure, such as two conformations or two stages of a
    motion. It deforms the initial state toward the final state and computes maps
    describing local strain and local rotation.

    The main outputs are file-based maps and Chimera scripts for visualization.
    The protocol produces converted maps for the initial state, final state,
    initial state deformed to the final state, strain, and local rotation. It also
    creates scripts to visualize strain, local rotation, and morphing between the
    initial and final maps.

    ## Inputs and General Workflow

    The protocol requires:

    - an initial-state volume;
    - a final-state volume;
    - a mask for the final state;
    - optionally, a symmetry group.

    The protocol runs a Matlab-based strain calculation using the Xmipp MIRT
    environment. The calculation deforms the initial state so that it fits the
    final state inside the region defined by the mask.

    The raw outputs are then converted to MRC format, mirrored as required by the
    coordinate convention, assigned the sampling rate of the input volumes, and
    optionally symmetrized. Finally, Chimera command scripts are generated for
    visualizing the strain map, local-rotation map, and morph between states.

    ## Initial State

    The **Initial state** parameter defines the starting volume.

    This is the volume that will be deformed toward the final state. It should
    represent the same structure as the final state, but in a different
    conformation or structural condition.

    The initial and final volumes must have the same box size. The protocol
    validates this requirement before running.

    ## Final State

    The **Final state** parameter defines the target volume.

    The strain calculation estimates how the initial state must deform to match
    this final state.

    The final state should be aligned with the initial state and should represent a
    comparable molecular region. If the volumes are not in the same coordinate
    frame, the estimated deformation, strain, and local rotation maps will be
    difficult to interpret.

    ## Mask for the Final State

    The **Mask for the final state** parameter defines where strain and local
    rotation are calculated.

    The mask should be a binary mask covering the relevant molecular region in the
    final-state map. Regions outside the mask are excluded from the deformation
    analysis.

    A good mask is important. If it is too tight, it may exclude relevant density.
    If it is too loose, background or solvent may influence the deformation
    calculation.

    ## Symmetry Group

    The **Symmetry group** parameter defines whether symmetry should be applied to
    the strain and local-rotation maps.

    The default is **c1**, meaning no symmetry.

    If the structure has known symmetry, the corresponding Xmipp symmetry group can
    be provided. After the strain and local-rotation maps are generated, the
    protocol can symmetrize these maps using the selected symmetry.

    Symmetry should only be used when it is biologically justified. Incorrect
    symmetry may average non-equivalent deformation features.

    ## Matlab/MIRT Calculation

    The core calculation is performed by a Matlab command that calls
    `xmipp_calculate_strain`.

    The calculation uses:

    - the final volume;
    - the initial volume;
    - the final-state mask;
    - an output file root.

    The Xmipp MIRT environment is used for the underlying deformation calculation.

    This step generates raw files that are later converted into MRC maps.

    ## Generated Maps

    The protocol prepares several maps from the raw calculation outputs:

    - `result_initial.mrc`;
    - `result_final.mrc`;
    - `result_initialDeformedToFinal.mrc`;
    - `result_strain.mrc`;
    - `result_localrot.mrc`.

    The initial and final maps are useful for checking the input states. The
    initial-deformed-to-final map helps inspect whether the deformation successfully
    brings the initial state toward the final state. The strain and local-rotation
    maps contain the main deformation descriptors.

    ## Strain Map

    The strain map describes local deformation magnitude between the initial and
    final states.

    Regions with stronger strain correspond to parts of the structure that undergo
    larger local deformation during the transformation from the initial state to
    the final state.

    This map can help identify flexible domains, hinge regions, interfaces under
    mechanical distortion, or areas involved in conformational change.

    The strain map should be interpreted as a deformation descriptor derived from
    the volume-to-volume registration, not as a direct physical measurement of
    force or energy.

    ## Local-Rotation Map

    The local-rotation map describes local rotational behavior in the deformation
    field.

    Regions with stronger local rotation correspond to areas where the deformation
    involves more rotational motion.

    This can help identify domains or subregions that rotate relative to the rest
    of the structure during the conformational transition.

    As with strain, local rotation should be interpreted in relation to map quality,
    masking, alignment, and biological context.

    ## Symmetrized Strain and Local-Rotation Maps

    If a symmetry group other than **c1** is selected, the protocol symmetrizes the
    strain and local-rotation maps.

    The symmetrization is performed without wrapping density across box
    boundaries. After symmetrization, the sampling rate is assigned again.

    This can make the deformation descriptors consistent with the symmetry of the
    structure, but only if that symmetry is appropriate for the conformational
    change being analyzed.

    ## Chimera Strain Script

    The protocol creates a Chimera command script named:

    `result_strain_chimera.cmd`

    This script opens the final-state map and the strain map, hides the strain
    volume, and colors the final map according to the strain values using a rainbow
    color scale with reversed colors.

    This provides a convenient way to visualize where strain is concentrated on
    the final-state structure.

    ## Chimera Local-Rotation Script

    The protocol creates a second Chimera command script named:

    `result_localrot_chimera.cmd`

    This script opens the final-state map and the local-rotation map, hides the
    local-rotation volume, and colors the final map according to local rotation.

    This helps visualize regions undergoing stronger rotational deformation.

    ## Chimera Morph Script

    The protocol also creates a morphing script named:

    `result_morph_chimera.cmd`

    This script opens the initial and final maps, hides them, and creates a morph
    between the two volumes using 50 frames.

    This visualization helps the user inspect the overall transition from the
    initial state to the final state.

    ## Output Behavior

    This protocol mainly produces file-based outputs rather than registered
    Scipion output volume objects.

    The important generated files are stored in the protocol extra folder and the
    Chimera scripts are written in the protocol working directory.

    Users should inspect the generated MRC files and visualization scripts to
    analyze the results.

    ## Validation Rules

    The protocol checks that the initial and final volumes have the same box size.

    If the box sizes differ, the protocol reports an error asking the user to make
    sure that the two volumes have the same size.

    The user should also ensure, even though not explicitly validated, that the
    volumes have compatible sampling rate, origin, alignment, and coordinate frame.

    ## Interpreting the Results

    The strain and local-rotation maps depend on the quality of the deformation
    from the initial state to the final state.

    Meaningful interpretation requires that the two volumes represent related
    states of the same structure, are aligned, have comparable resolution and
    sampling, and are analyzed with an appropriate mask.

    High strain or local rotation may indicate real conformational changes, but it
    may also arise from noise, misalignment, map artifacts, poor masking, or
    differences in local resolution.

    The deformed-initial map should be inspected to check whether the deformation
    reasonably matches the final state.

    ## Practical Recommendations

    Use this protocol for two related conformational states of the same structure.

    Make sure the initial and final maps have the same box size and are already
    aligned.

    Use a mask that covers the region where the deformation should be analyzed.

    Start with symmetry **c1** unless the deformation itself is expected to respect
    the structural symmetry.

    Inspect `result_initialDeformedToFinal.mrc` to verify that the deformation is
    reasonable before interpreting strain and local rotation.

    Use the generated Chimera scripts to visualize strain, local rotation, and the
    morph between states.

    Interpret high-strain or high-rotation regions together with the original
    density maps, local resolution, and biological knowledge.

    ## Final Perspective

    Calculate Strain is a deformation-analysis protocol for comparing two volume
    states.

    For biological users, its main value is that it converts a map-to-map
    conformational change into spatial descriptors of local strain and local
    rotation. These descriptors can help identify which regions deform, bend, or
    rotate between two structural states.

    The protocol should be used as an exploratory structural-analysis tool. Its
    results are most meaningful when the input maps are well aligned, comparable in
    quality, and interpreted together with the original volumes and biological
    context.
    """
    _label = 'calculate strain'
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addParam('inputVolume0', PointerParam, label="Initial state", important=True,
                      pointerClass='Volume',
                      help='Initial state of the structure, it will be deformed to fit into the final state')
        form.addParam('inputVolumeF', PointerParam, label="Final state", important=True,
                      pointerClass='Volume',
                      help='Initial state of the structure, it will be deformed to fit into the final state')
        form.addParam('inputMask', PointerParam, label="Mask for the final state", important=True,
                      pointerClass='VolumeMask',
                      help='Binary mask that defines where the strains and rotations will be calculated')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _getFileName(self, fnRoot, key, **kwargs):
        return "%s_%s.mrc"%(fnRoot,key)

    def _insertAllSteps(self):
        fnVol0 = self.inputVolume0.get().getFileName()
        fnVolF = self.inputVolumeF.get().getFileName()
        fnMask = self.inputMask.get().getFileName()
        self._insertFunctionStep(self.calculateStrain,fnVol0,fnVolF,fnMask)
        self._insertFunctionStep(self.prepareOutput)
        self._insertFunctionStep(self.createChimeraScript)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def calculateStrain(self, fnVol0, fnVolF, fnMask):
        fnRoot=self._getExtraPath('result')
        mirtDir = getXmippPath('external', 'mirt')
        # -wait -nodesktop
        args=('''-r "diary('%s'); xmipp_calculate_strain('%s','%s','%s','%s'); exit"'''
              % (fnRoot+"_matlab.log",fnVolF,fnVol0,fnMask,fnRoot))
        self.runJob("matlab", args, env=Plugin.getMatlabEnviron(mirtDir))
    
    def prepareOutput(self):
        volDim = self.inputVolume0.get().getDim()[0]
        Ts=self.inputVolume0.get().getSamplingRate()
        fnRoot=self._getExtraPath('result')

        def symmetrize(key):
            self.runJob("xmipp_transform_symmetrize", "-i %s --sym %s --dont_wrap" % \
                        (self._getFileName(fnRoot, key), self.symmetryGroup.get()))

        def changeSamplingRate(key):
            self.runJob("xmipp_image_header", "-i %s --sampling_rate %f" % (self._getFileName(fnRoot, key), Ts))

        def convert(key):
            self.runJob("xmipp_image_convert", "-i %s_%s.raw#%d,%d,%d,0,float -o %s" %
                        (fnRoot, key, volDim, volDim, volDim, self._getFileName(fnRoot, key)))
            self.runJob("xmipp_transform_mirror", "-i %s --flipX" % self._getFileName(fnRoot, key))
            changeSamplingRate(key)

        convert("initial")
        convert("final")
        convert("initialDeformedToFinal")
        convert("strain")
        convert("localrot")

        self.runJob("rm","-f "+self._getExtraPath('result_*.raw'))
        if self.symmetryGroup!="c1":
            symmetrize("strain")
            symmetrize("localrot")
            changeSamplingRate("strain")
            changeSamplingRate("localrot")

    def createChimeraScript(self):
        fnRoot = "extra/result"
        scriptFile = self._getPath('result') + '_strain_chimera.cmd'

        openStr = "open %s\n"

        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"strain"))
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        scriptFile = self._getPath('result') + '_localrot_chimera.cmd'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"localrot"))
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        scriptFile = self._getPath('result') + '_morph_chimera.cmd'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"initial"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write("vol #0 hide\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("vop morph #0,1 frames 50\n")
        fhCmd.close()

    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        xdim0 = self.inputVolume0.get().getDim()[0]
        xdimF = self.inputVolumeF.get().getDim()[0]
        if xdim0 != xdimF:
            errors.append("Make sure that the two volumes have the same size")
        return errors    
