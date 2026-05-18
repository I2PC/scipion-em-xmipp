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
from os.path import exists, join


from pyworkflow.object import String, Integer
from pyworkflow.protocol import BooleanParam, LEVEL_ADVANCED
from pyworkflow.utils.path import *

from pwem import EMObject
from pwem.protocols import ProtParticlePicking
from pwem.viewers import launchSupervisedPickerGUI

from pwem import emlib
from xmipp3.base import XmippProtocol
from xmipp3.convert import readSetOfCoordinates


class XmippProtParticlePicking(ProtParticlePicking, XmippProtocol):
    """ Picks particles in a set of micrographs
    either manually or in a supervised mode.

    AI Generated

    ## Overview

    The Manual Picking protocol allows the user to select particle coordinates on
    a set of micrographs, either manually or in supervised mode.

    Particle picking is the step where the approximate positions of particles are
    identified in the micrographs. These coordinates are later used by extraction
    protocols to cut out particle images for classification, alignment, and
    reconstruction.

    This protocol launches the Xmipp supervised particle-picking graphical
    interface from within Scipion. The user can inspect micrographs, define the
    particle size, pick particles, correct coordinates, and optionally use
    supervised or automatic assistance depending on the picking session.

    The main output is a set of particle coordinates associated with the input
    micrographs. The protocol can also save manually discarded coordinates if the
    user requests it.

    ## Inputs and General Workflow

    The input is a set of micrographs.

    When the protocol runs, it opens the particle-picking graphical interface. The
    user interacts with the micrographs and creates coordinate files. These
    coordinate files are stored in the protocol working directory.

    Once coordinates are available, the protocol creates a Scipion
    **SetOfCoordinates** linked to the input micrographs. It also reports the box
    size or particle size used during the picking session.

    If the option to save discarded particles is enabled, the protocol can also
    create an output set containing coordinates that were manually discarded during
    the picking process.

    ## Input Micrographs

    The protocol works on a set of input micrographs.

    These micrographs should normally be motion-corrected and suitable for visual
    inspection. In most workflows, CTF estimation may be performed before or after
    picking depending on the processing strategy, but the micrographs used for
    manual picking should have enough contrast for the user to recognize particles.

    The quality of particle picking strongly depends on the quality of the
    micrographs. Contaminated areas, carbon edges, crystalline ice, strong drift,
    or very low contrast may lead to incorrect coordinates and should be avoided
    during picking.

    ## Interactive Picking Interface

    The protocol launches the Xmipp supervised picker interface.

    In this interface, the user can inspect micrographs and select particles. The
    user can also correct mistakes, reject bad picks, and adjust the picking
    session according to the appearance of the data.

    The picking interface stores information such as the particle size, the current
    picking state, the number of manually picked particles, the number of
    automatically picked particles when applicable, and the last active micrograph.

    This makes the protocol useful even before final output coordinates are
    created, because the summary can report the current state of the picking
    session.

    ## Manual and Supervised Picking

    The protocol supports manual picking and supervised picking.

    In manual picking, the user directly selects particle positions. This is useful
    for small datasets, for training examples, for difficult specimens, or when the
    user wants complete control over the selected coordinates.

    In supervised picking, the user may provide examples and use automatic
    assistance to identify additional particles. This can accelerate picking while
    still allowing user supervision and correction.

    From a biological point of view, the goal is to select coordinates that
    represent true particles and avoid contaminants, aggregates, damaged particles,
    ice artifacts, and background features.

    ## Particle Size and Box Size

    During picking, the user defines a particle size. This value is later stored
    and reported by the protocol as the picking box size.

    The particle size should approximately match the apparent diameter of the
    particle in the micrograph. It is important because it guides visual selection
    and may be used by later extraction protocols to suggest an extraction box
    size.

    If the particle size is too small, the picking may focus only on part of the
    particle. If it is too large, nearby particles or background features may be
    included in the visual picking region.

    The particle size defined here is not necessarily the final extraction box
    size. Extraction often uses a somewhat larger box to include the particle plus
    surrounding background.

    ## Save Discarded Particles

    The **Save discarded particles** option creates an additional output containing
    coordinates that were manually discarded.

    This can be useful for quality-control, training, or method-development
    workflows. For example, discarded coordinates may represent contaminants,
    false positives, bad particles, or examples that should not be extracted.

    In routine biological processing, users may not need this output. However, it
    can be valuable when building training sets for automatic picking or when
    documenting why certain candidate particles were rejected.

    The discarded-coordinate output is only generated when there is a valid
    coordinate output and discarded coordinates are available.

    ## Run in Interactive Mode

    The **Run in interactive mode** option controls whether the protocol remains
    available for interactive picking sessions.

    If enabled, the user can pick particles across different sessions. This is the
    normal behavior for manual or supervised picking, because particle selection is
    often iterative.

    If disabled, the protocol finishes once an output coordinate set is created.
    This can be useful in scheduled or automated workflows where another protocol
    is waiting for the picking protocol to finish.

    Most users should keep interactive mode enabled during manual picking.

    ## Output Coordinates

    The main output is **outputCoordinates**, a set of particle coordinates linked
    to the input micrographs.

    Each coordinate identifies the position of a selected particle in its
    corresponding micrograph. These coordinates are normally passed to an extraction
    protocol to generate particle images.

    The output coordinate set also stores the picking box size. This value can be
    used by later protocols or wizards to suggest appropriate extraction settings.

    The coordinate output should be visually inspected before extraction,
    especially when using supervised or automatic assistance.

    ## Output Box Size

    The protocol also produces a **boxsize** output.

    This output stores the particle size used during picking. It is useful because
    the picking size is often needed later to choose an extraction box size.

    For example, an extraction protocol may suggest a box size based on the
    particle size from picking, often using a larger value to include surrounding
    background.

    ## Temporary Summary Before Final Output

    If the final coordinate output has not yet been created, the protocol can still
    show a temporary summary based on the picking configuration file.

    This summary may include:

    - the number of manually picked particles;
    - the particle size;
    - whether automatic picking was used;
    - the number of automatically picked particles;
    - the last micrograph visited.

    This is useful during interactive work because the user can monitor progress
    before finalizing the picking output.

    ## Practical Recommendations

    Use clean, representative micrographs for the initial picking session. Avoid
    micrographs with strong contamination, poor ice, severe drift, or very low
    contrast when defining examples.

    Choose a particle size that matches the apparent particle diameter. This value
    will influence later box-size suggestions.

    Manually inspect picks even when supervised or automatic assistance is used.
    False positives at this stage can propagate into particle extraction and
    classification.

    Consider saving discarded particles if you are developing automatic picking
    models, preparing training examples, or documenting rejected particle-like
    objects.

    Use interactive mode for normal manual picking. Disable it only when the
    protocol is part of a scheduled workflow that should finish automatically after
    coordinates are produced.

    After picking, continue with an extraction protocol to generate particle images
    from the selected coordinates.

    ## Final Perspective

    Manual Picking is the coordinate-generation step that connects micrograph
    inspection with particle extraction.

    For biological users, this protocol is important because the quality of the
    selected coordinates directly affects all downstream processing. Good picking
    provides true particle images for classification and reconstruction. Poor
    picking introduces contaminants, background, aggregates, or damaged particles
    that can reduce the quality of the final result.

    The protocol is especially useful when expert visual judgment is needed, when
    training supervised picking, or when preparing reliable initial coordinates for
    a new specimen.
    """
    _label = 'manual-picking (step 1)'

    def __init__(self, **args):        
        ProtParticlePicking.__init__(self, **args)
        # The following attribute is only for testing
        self.importFolder = String(args.get('importFolder', None))

    #--------------------------- DEFINE param functions ------------------------   
    def _defineParams(self, form):
        ProtParticlePicking._defineParams(self, form)

        form.addParam('saveDiscarded', BooleanParam, default=False,
                      label='Save discarded particles',
                      help='Generates an output with '
                           'the manually discarded particles.')
        form.addParam('doInteractive', BooleanParam, default=True,
                      label='Run in interactive mode',
                      expertLevel=LEVEL_ADVANCED,
                      help='If YES, you can pick particles in differents sessions.\n'
                           'If NO, once an outputCoordinates is created, '
                           'the protocol finishes. \n'
                           '(the last can be useful when other protocol '
                           'waits until this finish -internal scheduled-)')
              
    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        """The Particle Picking process is realized for a set of micrographs"""
        # Get pointer to input micrographs
        self.inputMics = self.inputMicrographs.get()
        micFn = self.inputMics.getFileName()

        # Launch Particle Picking GUI
        if not self.importFolder.hasValue():
            self._insertFunctionStep('launchParticlePickGUIStep', micFn,
                                      interactive=self.doInteractive)
        else: # This is only used for test purposes
            self._insertFunctionStep('_importFromFolderStep')
            # Insert step to create output objects
            self._insertFunctionStep('createOutputStep')

    def launchParticlePickGUIStep(self, micFn):
        # Launch the particle picking GUI
        extraDir = self._getExtraPath()
        process = launchSupervisedPickerGUI(micFn, extraDir, self)
        process.wait()
        # generate the discarded output only if there is a good output
        if self.saveDiscarded and exists(self._getPath('coordinates.sqlite')):
            self.createDiscardedStep()

        coordSet = self.getCoords()
        if coordSet:
            boxSize = Integer(coordSet.getBoxSize())
            self._defineOutputs(boxsize=boxSize)
            self._defineSourceRelation(self.inputMicrographs.get(), boxSize)

    def _importFromFolderStep(self):
        """ This function will copy Xmipp .pos files for
        simulating a particle picking run...this is only
        for testing purposes.
        """
        for f in getFiles(self.importFolder.get()):
            copyFile(f, self._getExtraPath())

    def createOutputStep(self):
        posDir = self._getExtraPath()
        coordSet = self._createSetOfCoordinates(self.inputMics)
        readSetOfCoordinates(posDir, self.inputMics, coordSet)
        self._defineOutputs(outputCoordinates=coordSet)
        self._defineSourceRelation(self.inputMicrographs, coordSet)

        boxSize = Integer(coordSet.getBoxSize())
        self._defineOutputs(boxsize=boxSize)
        self._defineSourceRelation(self.inputMicrographs.get(), boxSize)

    def createDiscardedStep(self):
        posDir = self._getExtraPath()
        suffixRoot = self._ProtParticlePicking__getOutputSuffix()
        suffix = '' if suffixRoot=='2' or suffixRoot=='' \
                 else str(int(suffixRoot)-1)
        coordSetDisc = self._createSetOfCoordinates(self.inputMics,
                                                    suffix='Discarded'+suffix)
        readSetOfCoordinates(posDir, self.inputMics, coordSetDisc,
                             readDiscarded=True)
        if coordSetDisc.getSize()>0:
            outputName = 'outputDiscardedCoordinates' + suffix
            outputs = {outputName: coordSetDisc}
            self._defineOutputs(**outputs)
            self._defineSourceRelation(self.inputMicrographs, coordSetDisc)
        
    #--------------------------- INFO functions --------------------------------
    def _citations(self):
        return ['Abrishami2013']

    #--------------------------- UTILS functions -------------------------------
    def __str__(self):
        """ String representation of a Supervised Picking run """
        if not hasattr(self, 'outputCoordinates'):
            msg = "No particles picked yet."
        else:

            picked = 0
            # Get the number of picked particles of the last coordinates set
            for key, output in self.iterOutputAttributes(EMObject):
                picked = output.getSize()

            msg = "%d particles picked (from %d micrographs)" % \
                  (picked, self.inputMicrographs.get().getSize())
    
        return msg

    def _methods(self):
        if self.getOutputsSize() > 0:
            return ProtParticlePicking._methods(self)
        else:
            return [self._getTmpMethods()]
    
    def _getTmpMethods(self):
        """ Return the message when there is not output generated yet.
         We will read the Xmipp .pos files and other configuration files.
        """
        configfile = join(self._getExtraPath(), 'config.xmd')
        existsConfig = exists(configfile)
        msg = ''
        
        if existsConfig:
            md = emlib.MetaData('properties@' + configfile)
            configobj = md.firstObject()
            pickingState = md.getValue(emlib.MDL_PICKING_STATE, configobj)
            particleSize = md.getValue(emlib.MDL_PICKING_PARTICLE_SIZE, configobj)
            isAutopick = pickingState != "Manual"
            manualParts = md.getValue(emlib.MDL_PICKING_MANUALPARTICLES_SIZE, configobj)
            autoParts = md.getValue(emlib.MDL_PICKING_AUTOPARTICLES_SIZE, configobj)

            if manualParts is None:
                manualParts = 0

            if autoParts is None:
                autoParts = 0

            msg = 'User picked %d particles ' % (autoParts + manualParts)
            msg += 'with a particle size of %d.' % particleSize

            if isAutopick:
                msg += "Automatic picking was used ([Abrishami2013]). "
                msg += "%d particles were picked automatically " %  autoParts
                msg += "and %d  manually." % manualParts

        return msg

    def _summary(self):
        if self.getOutputsSize() > 0:
            return ProtParticlePicking._summary(self)
        else:
            return [self._getTmpSummary()]

    def _getTmpSummary(self):
        summary = []
        configfile = join(self._getExtraPath(), 'config.xmd')
        existsConfig = exists(configfile)
        if existsConfig:
            md = emlib.MetaData('properties@' + configfile)
            configobj = md.firstObject()
            pickingState = md.getValue(emlib.MDL_PICKING_STATE, configobj)
            particleSize = md.getValue(emlib.MDL_PICKING_PARTICLE_SIZE, configobj)
            activeMic = md.getValue(emlib.MDL_MICROGRAPH, configobj)
            isAutopick = pickingState != "Manual"
            manualParticlesSize = md.getValue(emlib.MDL_PICKING_MANUALPARTICLES_SIZE, configobj)
            autoParticlesSize = md.getValue(emlib.MDL_PICKING_AUTOPARTICLES_SIZE, configobj)

            summary.append("Manual particles picked: %d"%manualParticlesSize)
            summary.append("Particle size:%d" %(particleSize))
            autopick = "Yes" if isAutopick else "No"
            summary.append("Autopick: " + autopick)
            if isAutopick:
                summary.append("Automatic particles picked: %d"%autoParticlesSize)
            summary.append("Last micrograph: " + activeMic)
        return "\n".join(summary)

    def getCoordsDir(self):
        return self._getExtraPath()
