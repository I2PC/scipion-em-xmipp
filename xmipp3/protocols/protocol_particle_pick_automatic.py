# **************************************************************************
# *
# * Authors:     Jose Gutierrez Tabuenca (jose.gutierrez@cnb.csic.es)
# *              Laura del Cano (laura.cano@cnb.csic.es)
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
from os.path import exists, basename, join

from pyworkflow.protocol.params import STEPS_PARALLEL, PointerParam, EnumParam, FileParam
from pyworkflow.utils.path import *

from pwem.protocols import ProtParticlePickingAuto

from pwem import emlib
from xmipp3.base import XmippProtocol
from xmipp3.convert import readSetOfCoordinates
from pyworkflow import BETA, UPDATED, NEW, PROD


MICS_SAMEASPICKING = 0
MICS_OTHER = 1
SRC_MANUAL_PICKING = 0
SRC_DIR = 1


class XmippParticlePickingAutomatic(ProtParticlePickingAuto, XmippProtocol):
    """Automatically picks particles from a set of micrographs using a previously trained model. This protocol speeds up particle selection by identifying particles consistently without manual intervention, improving throughput."""
    _label = 'auto-picking (step 2)'  
    _devStatus = PROD
    filesToCopy = ['model_svm.txt', 'model_pca_model.stk', 'model_rotpca_model.stk',
                   'model_particle_avg.xmp', 'templates.stk']
    
    def __init__(self, **kwargs):
        ProtParticlePickingAuto.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL
    
    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
    
        form.addSection(label='Input')

        form.addParam('modelSource', EnumParam, label="Model source",
                      choices=["Manual picking in this project", "External directory"],
                      default=0, help="The files model_* can be copied from a previous protocol execution within this "
                                      "project or copied from an external directory. This latter option is useful in"
                                      "cases in which the same kind of molecule is processed many times.")

        form.addParam('xmippParticlePicking', PointerParam,
                      label="Xmipp particle picking run",
                      pointerClass='XmippProtParticlePicking',
                      condition="modelSource==%d"%SRC_MANUAL_PICKING,
                      #pointerCondition='isFinished',
                      help='Select the previous xmipp particle picking run.')

        form.addParam('xmippParticlePickingDir', FileParam,
                      label="Xmipp particle picking model directory",
                      allowsNull=True,
                      condition="modelSource==%d"%SRC_DIR,
                      #pointerCondition='isFinished',
                      help='The directory must contain the files model_*, config.xmd and templates.stk')

        form.addParam('micsToPick', EnumParam,
                      choices=['Same as supervised', 'Other'],
                      default=0, label='Micrographs to pick',
                      display=EnumParam.DISPLAY_LIST,
                      help="Select from which set of micrographs to pick using "
                           "the training from supervised run."
                           "If you use Same as supervised, the same set of "
                           "micrographs used for training the picker will be "
                           "used at this point. If you select Other, you can "
                           "select another set of micrograph (normally from "
                           "the same specimen) and pick them completely "
                           "automatic using the trained picker.")

        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs',
                      condition='micsToPick==%d' % MICS_OTHER,
                      label="Micrographs",
                      help="Select other set of micrographs to pick using the "
                           "trained picker.")

        self._defineStreamingParams(form)

        form.addParallelSection(threads=1, mpi=1)
        
    # --------------------------- INSERT steps functions -----------------------
    def _insertInitialSteps(self):
        # Get pointer to input micrographs
        self.particlePickingRun = self.xmippParticlePicking.get()

        copyId = self._insertFunctionStep('copyInputFilesStep')

        return [copyId]

    # --------------------------- STEPS functions ------------------------------
    def getSrcDir(self):
        if self.modelSource == SRC_MANUAL_PICKING:
            return self.xmippParticlePicking.get()._getExtraPath()
        else:
            return self.xmippParticlePickingDir.get()

    def copyInputFilesStep(self):
        # Copy training model files to current run
        srcDir = self.getSrcDir()
        for f in self.filesToCopy:
            createLink(os.path.join(srcDir, f), self._getExtraPath(f))
        copyFile(os.path.join(srcDir, "config.xmd"), self._getExtraPath("config.xmd"))

        # Get the box size
        mdInfo = emlib.MetaData("properties@"+self._getExtraPath("config.xmd"))
        self.boxSize = mdInfo.getValue(emlib.MDL_PICKING_PARTICLE_SIZE,mdInfo.firstObject())
        mdInfo.setValue(emlib.MDL_PICKING_MANUALPARTICLES_SIZE,0,mdInfo.firstObject())
        mdInfo.write("properties@"+self._getExtraPath("config.xmd"),emlib.MD_APPEND)

    def _pickMicrograph(self, mic, *args):
        micPath = mic.getFileName()
        # Get particle picking boxsize from the previous run
        modelRoot = self._getExtraPath('model')

        micName = removeBaseExt(micPath)
        proceed = True
        if self.micsToPick == MICS_SAMEASPICKING:
            basePos = replaceBaseExt(micPath, "pos")
            fnPos = self.particlePickingRun._getExtraPath(basePos)
            if exists(fnPos):
                blocks = emlib.getBlocksInMetaDataFile(fnPos)
                copy = True
                if 'header' in blocks:
                    mdheader = emlib.MetaData("header@" + fnPos)
                    state = mdheader.getValue(emlib.MDL_PICKING_MICROGRAPH_STATE,
                                              mdheader.firstObject())
                    if state == "Available":
                        copy = False
                if copy:
                    # Copy manual .pos file of this micrograph
                    copyFile(fnPos, self._getExtraPath(basename(fnPos)))
                    proceed = False            

        if proceed:
            args = "-i %s " % micPath
            args += "--particleSize %d " % self.boxSize
            args += "--model %s " % modelRoot
            args += "--outputRoot %s " % self._getExtraPath(micName)
            args += "--mode autoselect --thr %d" % self.numberOfThreads

            self.runJob("xmipp_micrograph_automatic_picking", args)

    def readSetOfCoordinates(self, workingDir, coordSet):
        readSetOfCoordinates(workingDir, self.getInputMicrographs(), coordSet)

    def readCoordsFromMics(self, workingDir, micList, coordSet):
        readSetOfCoordinates(workingDir, micList, coordSet)
        
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        validateMsgs = []

        if self.modelSource == SRC_MANUAL_PICKING and not hasattr(self.xmippParticlePicking.get(),"outputCoordinates"):
            validateMsgs.append("You need to generate coordinates for the "
                                "supervised picking")
   
        srcDir = self.getSrcDir()
        srcPaths = [os.path.join(srcDir,k) for k in self.filesToCopy]
        # Check that all needed files exist
        if missingPaths(*srcPaths):
            validateMsgs.append('Input picking run has not been trained, '
                                'use *Autopick* for at least one micrograph')
            
        # If other set of micrographs is provided they should have same
        # sampling rate and acquisition
        if self.micsToPick.get() == MICS_OTHER and self.modelSource != SRC_DIR:
            inputMics = self.inputMicrographs.get()
            manualMics = self.xmippParticlePicking.get().inputMicrographs.get()
            # FIXME: manualMics is always None when scheduled...
            #  it should be fixed in the update step at Scipion scheduler app
            if manualMics is not None:
                pixsizeInput = inputMics.getSamplingRate()

                pixsizeMics = manualMics.getSamplingRate()
                acq = manualMics.getAcquisition()

                if pixsizeInput != pixsizeMics:
                    validateMsgs.append('New micrographs should have same sampling '
                                        'rate as the ones already picked.')

                if not inputMics.getAcquisition().equalAttributes(acq):
                    validateMsgs.append('New micrographs should have same '
                                        'acquisition parameters as the ones '
                                        'already picked.')

        if self.modelSource.get()==SRC_DIR and self.micsToPick.get()==MICS_SAMEASPICKING:
            validateMsgs.append("You cannot take the model from a directory and indicate that the set of micrograohs "
                                "is the same as picking. If you take the model from a directory, probably you want "
                                "to pick from a different set.")
        return validateMsgs
    
    def getSummary(self, coordSet):
        summary = []
        if self.modelSource == SRC_MANUAL_PICKING:
            summary.append("Previous run: %s" %
                           self.xmippParticlePicking.get().getNameId())
        else:
            summary.append("Model from: %s" %
                           self.xmippParticlePickingDir.get())
        return "\n".join(summary)

    def getMethods(self, output):
        manualPickName = self.xmippParticlePicking.get().getNameId()
        msg = 'Program picked %d particles ' % output.getSize()
        msg += 'of size %d ' % output.getBoxSize()
        msg += 'using training from %s. ' % manualPickName
        msg += 'For more detail see [Abrishami2013]'
        return msg

    def _citations(self):
        return ['Abrishami2013']

    # --------------------------- UTILS functions ------------------------------
    def getCoordsDir(self):
        return self._getExtraPath()
    
    def getInputMicrographsPointer(self):
        # Get micrographs to pick
        if self.micsToPick == MICS_SAMEASPICKING:
            inputPicking = self.xmippParticlePicking.get()
            return inputPicking.inputMicrographs if inputPicking else None
        else:
            return self.inputMicrographs
        
    def getInputMicrographs(self):
        """ Return the input micrographs that can be the same of the supervised
        picking or other ones selected by the user. (This can be used to pick
        a new set of micrographs with the same properties than a previous
        trained ones. )
        """ 
        return self.getInputMicrographsPointer().get() if self.getInputMicrographsPointer() else None
