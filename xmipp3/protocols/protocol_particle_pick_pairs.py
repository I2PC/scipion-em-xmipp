# **************************************************************************
# *
# * Authors:     Jose Gutierrez Tabuenca (jose.gutierrez@cnb.csic.es)
# *              J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

from pyworkflow.object import String
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pwem.protocols import ProtParticlePicking
from pwem.objects import CoordinatesTiltPair
from pwem.viewers import launchTiltPairPickerGUI

from xmipp3 import convert
from xmipp3.base import XmippProtocol


class XmippProtParticlePickingPairs(ProtParticlePicking, XmippProtocol):
    """ Picks particles in paired tilted micrographs. Using paired data
    improves particle localization and orientation determination, enhancing
    reconstruction accuracy.

    AI Generated

    ## Overview

    The Tilt Pairs Particle Picking protocol allows the user to pick corresponding
    particles in paired tilted and untilted micrographs.

    Tilt-pair data contain two related images of the same field: one acquired
    without tilt, or with lower tilt, and another acquired at a known tilt angle.
    The same particles appear in both micrographs, but their positions are related
    by the geometry of the tilt. Correctly identifying corresponding particles in
    both views is essential for workflows such as random conical tilt,
    tilted-pair validation, and geometry-aware reconstruction strategies.

    This protocol launches the Xmipp tilt-pair particle-picking graphical
    interface. The user can inspect the paired micrographs and define particle
    coordinates in a way that preserves the relationship between untilted and
    tilted views.

    The main output is a **CoordinatesTiltPair** object, containing linked
    coordinate sets for the untilted and tilted micrographs, together with the
    tilt-angle information read from the paired micrograph metadata.

    ## Inputs and General Workflow

    The input is a set of paired tilted micrographs, represented in Scipion as a
    **MicrographsTiltPair** object.

    The protocol first converts the paired micrograph information into Xmipp
    metadata format. This metadata describes the relationship between the untilted
    and tilted micrograph sets and is used by the graphical picker.

    The protocol then opens the tilt-pair picking interface. After the user has
    picked particles and saved the results, the protocol reads the coordinate files
    for both the untilted and tilted micrographs. It also reads the tilt-pair angle
    information and creates the final paired-coordinate output.

    ## Micrographs Tilt Pair

    The **Micrographs tilt pair** parameter defines the paired micrograph dataset
    to be used for picking.

    This input contains two linked micrograph sets:

    - the untilted micrographs;
    - the tilted micrographs.

    Each untilted micrograph should correspond to a tilted micrograph showing the
    same field of view under the tilt geometry. The quality of the output depends
    on this pairing being correct.

    If the micrograph pairs are mismatched, the picked coordinates will not
    represent the same physical particles in both views, and downstream tilted-pair
    analysis will be unreliable.

    ## Tilt-Pair Picking Interface

    The protocol launches a graphical interface designed specifically for
    tilt-pair picking.

    Unlike ordinary single-micrograph picking, the goal here is not only to select
    particle positions, but also to preserve their correspondence between the two
    views. The picker uses the paired micrograph metadata to help manage this
    relationship.

    The user can inspect the untilted and tilted images, pick particles, and save
    coordinate information for both members of each pair.

    This interactive step is important because tilted images may be harder to
    interpret than untilted images. Particles can appear distorted, shifted, or
    less contrasted because of the tilt geometry and increased effective ice
    thickness.

    ## Untilited and Tilted Coordinate Sets

    After picking, the protocol creates two coordinate sets:

    - one coordinate set for the untilted micrographs;
    - one coordinate set for the tilted micrographs.

    These coordinate sets are not independent outputs. They are combined into a
    single **CoordinatesTiltPair** object that stores their relationship.

    This paired organization is essential. It tells downstream protocols which
    untilted coordinate corresponds to which tilted coordinate.

    ## Tilt Angles

    The protocol also reads the angle information associated with the input
    micrograph pairs.

    These angles describe the tilt geometry of the paired acquisition. They are
    stored together with the coordinate pairs and are needed by downstream
    tilted-pair processing protocols.

    The angle information is not manually estimated by this protocol. It is read
    from the paired micrograph metadata generated from the input
    MicrographsTiltPair object.

    Users should therefore ensure that the tilt-pair micrograph input contains
    correct tilt information.

    ## Output CoordinatesTiltPair

    The main output is an **outputCoordinatesTiltPair** object.

    This output contains:

    - the untilted coordinate set;
    - the tilted coordinate set;
    - the angle information linking the paired micrographs;
    - the box-size information when available.

    This object is the standard Scipion representation of picked tilt-pair
    coordinates. It can be used by protocols that extract paired particles or
    perform tilted-pair analysis.

    If the protocol is run several times and new coordinate outputs are registered,
    the outputs may receive numbered suffixes to avoid overwriting previous
    results.

    ## Particle Size and Box Size

    The protocol can store a particle box size in the output coordinate sets when
    available.

    The box size represents the approximate particle size used during picking and
    can guide later extraction of untilted and tilted particle pairs.

    For tilted-pair data, the box size should be large enough to contain the
    particle in both views. Tilted projections may appear slightly elongated,
    distorted, or less well centered, so users should be careful not to choose a
    box that is too small.

    ## Interpretation of the Output

    The output should be interpreted as a set of paired particle positions.

    Each pair represents one physical particle observed in two acquisition
    geometries. The untilted coordinate indicates its position in the untilted
    micrograph, and the tilted coordinate indicates the corresponding position in
    the tilted micrograph.

    This correspondence is more important than either coordinate set alone.
    Downstream analysis depends on the assumption that each pair refers to the same
    particle.

    ## Practical Recommendations

    Use this protocol only with correctly paired tilted and untilted micrographs.

    Before picking, inspect the micrograph pairs to verify that they correspond to
    the same field of view and that the tilt geometry is sensible.

    Pick particles that can be reliably identified in both views. Avoid ambiguous
    particles, overlapping particles, strong contamination, carbon edges, and areas
    where the tilted view is too degraded to identify the same particle.

    Choose a particle size or extraction box large enough for the tilted view, not
    only for the untilted view.

    After picking, inspect the paired coordinates before extraction. Errors in
    pairing can seriously affect downstream tilted-pair workflows.

    If several picking sessions are needed, keep track of the different output
    coordinate-pair objects generated by the protocol.

    ## Final Perspective

    Tilt Pairs Particle Picking is the coordinate-generation step for tilted-pair
    cryo-EM workflows.

    For biological users, its main role is to define which particles correspond
    between untilted and tilted micrographs. This correspondence is the foundation
    for subsequent extraction and analysis of particle pairs.

    Good tilted-pair picking requires both accurate particle identification and
    careful preservation of the pairing relationship. When done correctly, it
    provides the geometrical information needed for downstream protocols that use
    tilt-pair data.
    """
    _label = 'tilt pairs particle picking'

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        # The following attribute is only for testing
        self.importFolder = String(args.get('importFolder', None))

    #--------------- DEFINE param functions ---------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMicrographsTiltedPair', params.PointerParam,
                      pointerClass='MicrographsTiltPair',
                      label="Micrographs tilt pair",
                      help='Select the MicrographsTiltPair ')

        #----------- INSERT steps functions ----------------------------------
    def _insertAllSteps(self):
        """ The Particle Picking process is realized for a pair
        of set of micrographs
        """
        self.micsFn = self._getPath('input_micrographs.xmd')
        # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')

        # Launch Particle Picking GUI
        if not self.importFolder.hasValue():
            self._insertFunctionStep('launchParticlePickGUIStep', interactive=True)
        else: # This is only used for test purposes
            self._insertFunctionStep('_importFromFolderStep')

    #------------------- STEPS functions -----------------------------------
    def convertInputStep(self):
        micTiltPairs = self.inputMicrographsTiltedPair.get()
        # Get the converted input micrographs in Xmipp format
        convert.writeSetOfMicrographsPairs(micTiltPairs.getUntilted(),
                                           micTiltPairs.getTilted(),
                                           self.micsFn)

    def launchParticlePickGUIStep(self):
        process = launchTiltPairPickerGUI(self.micsFn, self._getExtraPath(), self)
        process.wait()

    def _importFromFolderStep(self):
        """ This function will copy Xmipp .pos files for
        simulating a particle picking run...this is only
        for testing purposes.
        """
        extraDir = self._getExtraPath()

        for f in pwutils.getFiles(self.importFolder.get()):
            pwutils.copyFile(f, extraDir)

        self.registerCoords(extraDir, readFromExtra=True)

    #--------------------------- INFO functions --------------------------------------------
    def _citations(self):
        return []

    #--------------------------- UTILS functions -------------------------------------------
    def __str__(self):
        """ String representation of a Particle Picking Tilt run """
        outputs = self.getOutputsSize()

        if outputs == 0:
            msg = "No particles picked yet."
        elif outputs == 1:
            picked = self.getCoords().getSize()
            mics = self.inputMicrographsTiltedPair.get().getTilted().getSize()
            msg = "Number of particles picked: %d " % picked
            msg += "(from %d micrographs)" % mics
        else:
            msg = 'Number of outputs: %d' % outputs

        return msg

    def getInputMicrographs(self):
        return self.inputMicrographsTiltedPair.get().getTilted()

    def getCoords(self):
        return self.getCoordsTiltPair()

    def _summary(self):
        summary = []
        if self.getInputMicrographs() is  not None:
            summary.append("Number of input micrographs: %d"
                           % self.getInputMicrographs().getSize())

        if self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes(CoordinatesTiltPair):
                summary.append("*%s:*" % key)
                summary.append("  Particles pairs picked: %d" % output.getSize())
                summary.append("  Particle size: %d \n" % output.getBoxSize())
        else:
            summary.append("Output tilpairs not ready yet.")

        return summary

    def __getOutputSuffix(self):
        maxCounter = -1
        for attrName, _ in self.iterOutputAttributes(CoordinatesTiltPair):
            suffix = attrName.replace('outputCoordinatesTiltPair', '')
            try:
                counter = int(suffix)
            except:
                counter = 1 # when there is not number assume 1
            maxCounter = max(counter, maxCounter)

        return str(maxCounter+1) if maxCounter > 0 else '' # empty if not outputs

    def _getBoxSize(self):
        """ Redefine this function to set a specific box size to the output
        coordinates untilted and tilted.
        """
        return None

    def _readCoordinates(self, coordsDir, suffix=''):
        micTiltPairs = self.inputMicrographsTiltedPair.get()
        uSuffix = 'Untilted' + suffix
        tSuffix = 'Tilted' + suffix
        uSet = micTiltPairs.getUntilted()
        tSet = micTiltPairs.getTilted()
        # Create Untilted and Tilted SetOfCoordinates
        uCoordSet = self._createSetOfCoordinates(uSet, suffix=uSuffix)
        convert.readSetOfCoordinates(coordsDir, uSet, uCoordSet)
        uCoordSet.write()
        tCoordSet = self._createSetOfCoordinates(tSet, suffix=tSuffix)
        convert.readSetOfCoordinates(coordsDir, tSet, tCoordSet)
        tCoordSet.write()
        boxSize = self._getBoxSize()
        if boxSize:
            uCoordSet.setBoxSize(boxSize)
            tCoordSet.setBoxSize(boxSize)

        return uCoordSet, tCoordSet

    def _readAngles(self, micsFn, suffix=''):
        # Read Angles from input micrographs
        anglesSet = self._createSetOfAngles(suffix=suffix)
        convert.readAnglesFromMicrographs(micsFn, anglesSet)
        anglesSet.write()
        return anglesSet

    def registerCoords(self, coordsDir, store=True, readFromExtra=False):
        micTiltPairs = self.inputMicrographsTiltedPair.get()
        suffix = self.__getOutputSuffix()

        uCoordSet, tCoordSet = self._readCoordinates(coordsDir, suffix)
        
        if readFromExtra:
            micsFn = self._getExtraPath('input_micrographs.xmd')
        else:
            micsFn = self._getPath('input_micrographs.xmd')
            
        anglesSet = self._readAngles(micsFn, suffix)
        # Create CoordinatesTiltPair object
        outputset = self._createCoordinatesTiltPair(micTiltPairs,
                                                    uCoordSet, tCoordSet,
                                                    anglesSet, suffix)
        summary = self.getSummary(outputset)
        outputset.setObjComment(summary)
        outputName = 'outputCoordinatesTiltPair' + suffix
        outputs = {outputName: outputset}
        self._defineOutputs(**outputs)
        self._defineSourceRelation(self.inputMicrographsTiltedPair, outputset)
        if store:
            self._store()

