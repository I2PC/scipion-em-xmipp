# **************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, IntParam
from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.protocols import EMProtocol
from pwem.objects import Volume


class XmippMetaProtCreateSubset(EMProtocol):
    """ Metaprotocol to select a set of particles from a 3DClasses and a
    Volume from a SetOfVolumes
     """
    _label = 'metaprotocol heterogeneity subset'
    _lastUpdateVersion = VERSION_2_0
    _devStatus = BETA

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        # form.addParam('inputSetOfVolumes', PointerParam, pointerClass='SetOfVolumes',
        #               label="Input set of volumes",
        #               help='Select the set of volumes to select an specific volume.')

        form.addParam('inputSetOfClasses3D', PointerParam, pointerClass='SetOfClasses3D',
                      label="Input set of 3D classes",
                      help='Select the set of 3D classes to select an specific set of particles.')

        form.addParam('idx', IntParam,
                      label="Input Idx", default=0,
                      help='Identifier of the volume and particles selected')

         
    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions -------------------------------

    def createOutputStep(self):

        id = self.idx.get()
        # inputSetOfVolumes=self.inputSetOfVolumes.get()
        inputSetOfClasses3D=self.inputSetOfClasses3D.get()

        newVolume = Volume()
        #newVolume.setFileName(inputSetOfVolumes[id].getFileName())
        #newVolume.setSamplingRate(inputSetOfVolumes[id].getSamplingRate())
        newVolume.setFileName(inputSetOfClasses3D[id].getRepresentative().getFileName())
        newVolume.setSamplingRate(inputSetOfClasses3D[id].getSamplingRate())

        newParticles = self._createSetOfParticles()
        newParticles.setSamplingRate(inputSetOfClasses3D[id].getSamplingRate())
        if inputSetOfClasses3D[id].hasAlignmentProj():
            newParticles.setAlignmentProj()
        newParticles.copyItems(inputSetOfClasses3D[id])
        self._defineOutputs(**{'outputAuxVolumes': newVolume})
        self._store(newVolume)
        self._defineOutputs(**{'outputAuxParticles': newParticles})
        self._store(newParticles)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        return summary


