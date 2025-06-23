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

    (AI Generated)

    Purpose and Scope. This protocol is used to manually or semi-automatically select particles from a set of
    micrographs, typically as the first step in a single-particle analysis workflow. It launches a graphical interface
    that allows the user to identify individual particles by clicking on them or to inspect results of supervised
    particle picking. It is designed for interactive use but can also run in a batch mode when needed for scheduled or
    automated workflows.

    Inputs. The input consists of a set of micrographs. These micrographs must have been previously imported or processed in Scipion and contain valid image paths. Optionally, the protocol allows the user to activate or deactivate interactive mode and to decide whether discarded particles (manually removed during picking) should be saved as a separate output.

Protocol Behavior

The protocol launches the Xmipp supervised particle picker graphical interface, which allows the user to inspect micrographs one by one and either manually pick particles or adjust parameters for automatic picking. During the session, picked particles are saved and, if configured, discarded particles are also tracked.

If the protocol is run in non-interactive mode, it finishes automatically after the first session, which is useful when this protocol is called as part of a larger workflow. If run in interactive mode, the user can perform multiple sessions of particle picking before finalizing the protocol.

For testing purposes or scripted runs, the protocol supports importing coordinates from a folder instead of launching the GUI. This is done by setting an internal option importFolder, not exposed in the graphical interface.

Outputs

The main output is a set of particle coordinates corresponding to the manually or automatically selected particles. These coordinates can be used as input to downstream steps such as particle extraction. If the user has chosen to save discarded particles, a separate output is generated containing those coordinates. The protocol also records the box size used during picking, which is typically derived from the picker configuration.

User Workflow

The user provides a set of micrographs and launches the protocol. The GUI opens, allowing manual selection or supervised picking. After completing the session, the protocol creates an output coordinate set and optionally a discarded coordinate set. These can be viewed in Scipion and passed to downstream protocols for further processing, such as extraction, classification, or refinement.

Interpretation and Best Practices

This protocol is ideal for initial particle selection, particularly when automated picking methods are not yet trained. It is especially useful in early stages of a project where representative particle views are needed to create templates or to perform initial classifications.

If a supervised picker is enabled in the GUI, automatic particle proposals may be shown and confirmed or rejected. The protocol can handle large numbers of micrographs but benefits from chunking the picking into multiple sessions to prevent fatigue and ensure quality control.

When working in scheduled workflows, it is advisable to disable interactive mode so that downstream protocols can run without interruption.
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
