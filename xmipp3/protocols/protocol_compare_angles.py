# **************************************************************************
# *
# * Authors:     C.O.S. Sorzano (coss@cnb.csic.es)
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

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_2_0
from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ

from pwem import emlib
from xmipp3.convert import (setXmippAttributes, writeSetOfParticles)
from xmipp3.constants import SYM_URL


class XmippProtCompareAngles(ProtAnalysis3D):
    """    
    Compare two sets of angles. The output is a list of all common particles
    with the angular difference between both assignments. The output is
    constructed by keeping the information from the Set 1 and adding the
    shiftDiff and angularDiff.

    This protocol answers a very practical question:

    “How different are the angular assignments (and shifts) of the same
    particles between two alignment/refinement results?”

    You give it two particle sets that already have projection alignment
    (rot/tilt/psi + shifts). It outputs a new particle set containing only the
    particles that exist in both inputs, with two extra per-particle measures:

    - angularDiff: the angular distance (in degrees) between the two assigned
    orientations
    - shiftDiff: the difference between the two in-plane shifts

    It keeps the “identity” and metadata of Set 1, and appends these comparison
    numbers.

    When is this useful?

    Common scenarios:

    - Compare two refinements / two pipelines
    - Same dataset refined with different settings (mask, solvent flattening,
    CTF refinement, high/low-pass choices, etc.).
    - You want to see if orientations are stable or if the solution “moved”.
    - Check for alignment instability / overfitting

    If many particles show large angularDiff, it can indicate:
    - poor SNR / bad particles,
    - ambiguous views,
    - reference bias or wrong initial model,
    - symmetry/mirror ambiguities.

    Quantify effect of symmetry choice

    You can compare runs with/without symmetry, or with different symmetry
    handling, using the appropriate symmetryGroup in this protocol.

    Compare “consensus” vs “per-class/per-iteration” assignments

    E.g., angles from a global refinement vs angles imported from another software.

    What you get out (and how you’d interpret it)
    Output

    A SetOfParticles (projection-aligned) containing:

    Only common particles (intersection by itemId)

    Two added attributes per particle:

    angleDiff

    shiftDiff

    Interpretation heuristics

    Small angularDiff for most particles → assignments are consistent; your
    refinement is stable.

    A tail of large angularDiff → subset of particles is unstable (often junk,
    low SNR, flexible regions, rare views).

    Large angularDiff cluster-wide → runs disagree globally (different minima,
    wrong symmetry, reference bias, different masking strategy, etc.).

    Small angularDiff but large shiftDiff → orientations agree but
    centering/translation differs (box center issues, different alignment
    constraints).

    A very practical follow-up is to plot histograms of angularDiff and
    shiftDiff, and/or to filter particles with large differences to inspect
    them (do they look like junk? do they concentrate in certain views?).

    The “symmetry group” parameter matters

    Angular differences are computed modulo symmetry using Xmipp conventions. In practice:

    If your particle is in C3, two orientations that differ by a 120° rotation
    around the symmetry axis should be considered equivalent.

    This protocol accounts for that by using the symmetryGroup you provide.

    It also enables --check_mirrors, so it tries to resolve mirror ambiguities
    when comparing assignments (useful when a solution might flip
    handedness/mirror-related assignments).

    Rule of thumb: set symmetryGroup to the same symmetry you used in
    refinement (or the one you want to compare under).

    What this protocol does not do

    It doesn’t “decide which assignment is correct”.

    It doesn’t modify the original angles to reconcile them.

    It doesn’t compare particles that are not common to both sets (it
    explicitly intersects by particle ID).

    It’s a diagnostic/QA tool.

    Typical practitioner workflow

    - Run two refinements / two angle assignment methods.
    - Use Compare Angles with the same symmetry assumption used in the run.
    - Inspect distribution of angularDiff and shiftDiff.
    Optionally:
    - exclude particles with large differences,
    - inspect those particles by class/view,
    - re-run refinement with improved cleaning or constraints.
    """

    _label = 'compare angles'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)
        
    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addParam('inputParticles1', params.PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles 1",  
                      help='Select the input experimental images with an '
                           'angular assignment.')

        form.addParam('inputParticles2', params.PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles 2",  
                      help='Select the input experimental images with an '
                           'angular assignment.')

        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group", 
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)

    
    #--------------------------- INSERT steps functions ------------------------

    def _insertAllSteps(self):        
        self._insertFunctionStep('convertInputStep',
                                 self.inputParticles1.get().getObjId(),
                                 self.inputParticles2.get().getObjId())

        self._insertFunctionStep('analyzeDistanceStep',
                                 self.inputParticles1.get().getObjId(),
                                 self.inputParticles2.get().getObjId(),
                                 self.symmetryGroup.get())

        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions -------------------------------

    def convertInputStep(self, particlesId1, particlesId2):
        """ Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        writeSetOfParticles(self.inputParticles1.get(), self._getExtraPath("angles1.xmd"), alignType=ALIGN_PROJ)
        writeSetOfParticles(self.inputParticles2.get(), self._getExtraPath("angles2.xmd"), alignType=ALIGN_PROJ)

    def analyzeDistanceStep(self, particlesId1, particlesId2, symmetryGroup):
        self.runJob("xmipp_metadata_utilities","-i %s -o %s --operate keep_column itemId"%\
                    (self._getExtraPath("angles1.xmd"),self._getTmpPath("ids1.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s -o %s --operate keep_column itemId"%\
                    (self._getExtraPath("angles2.xmd"),self._getTmpPath("ids2.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getTmpPath("ids1.xmd"),self._getTmpPath("ids2.xmd"),self._getTmpPath("ids.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getExtraPath("angles1.xmd"),self._getTmpPath("ids.xmd"),self._getExtraPath("angles1_common.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getExtraPath("angles2.xmd"),self._getTmpPath("ids.xmd"),self._getExtraPath("angles2_common.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --operate sort itemId"%self._getExtraPath("angles1_common.xmd"))
        self.runJob("xmipp_metadata_utilities","-i %s --operate sort itemId"%self._getExtraPath("angles2_common.xmd"))

        self.runJob("xmipp_angular_distance","--ang1 %s --ang2 %s --sym %s --check_mirrors --oroot %s"%\
                    (self._getExtraPath("angles1_common.xmd"),self._getExtraPath("angles2_common.xmd"),
                     self.symmetryGroup,self._getTmpPath("angular_distance")))
        self.runJob("xmipp_metadata_utilities",'-i %s -o %s --operate keep_column "angleDiff shiftDiff"'%\
                    (self._getTmpPath("angular_distance.xmd"),self._getTmpPath("diffs.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set merge %s"%\
                    (self._getExtraPath("angles1_common.xmd"),self._getTmpPath("diffs.xmd")))


    def createOutputStep(self):
        imgSet1 = self.inputParticles1.get()
        imgSetOut = self._createSetOfParticles()
        imgSetOut.copyInfo(imgSet1)
        imgSetOut.setAlignmentProj()
        self.iterMd = md.iterRows(self._getExtraPath("angles1_common.xmd"), md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd) 
        imgSetOut.copyItems(imgSet1,
                            updateItemCallback=self._updateItem)
        self._defineOutputs(outputParticles=imgSetOut)
        self._defineSourceRelation(self.inputParticles1, imgSetOut)
        self._defineSourceRelation(self.inputParticles2, imgSetOut)

    def _updateItem(self, particle, row):
        count = 0
        
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
                    
        particle._appendItem = count > 0
        
    def _createItemMatrix(self, particle, row):
        setXmippAttributes(particle, row, emlib.MDL_SHIFT_DIFF, emlib.MDL_ANGLE_DIFF)

    # --------------------------- INFO functions -------------------------------

    def _validate(self):
        validateMsgs = []
        if self.inputParticles1.get() and not self.inputParticles1.hasValue():
            validateMsgs.append('Please provide input particles 1.')            
        if self.inputParticles2.get() and not self.inputParticles2.hasValue():
            validateMsgs.append('Please provide input particles 2.')            
        return validateMsgs
    
    def _summary(self):
        summary = []
        return summary

