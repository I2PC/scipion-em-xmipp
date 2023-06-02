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

from shutil import copy

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import MultiPointerParam, PointerParam
from pwem.protocols import EMProtocol
from pwem.objects import Volume, Class3D


class XmippMetaProtCreateOutput(EMProtocol):
    """ Metaprotocol to run together all the protocols to discover discrete
    heterogeneity in a set of particles
     """
    _label = 'metaprotocol heterogeneity output'
    _lastUpdateVersion = VERSION_2_0

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMetaProt', PointerParam, important=True,
                      label="Input MetaProtocol", pointerClass='XmippMetaProtDiscreteHeterogeneity',
                      help='Select the metaprotocol heterogeneity '
                           'to create the final setOfVolumes and setOfClasses3D.')

        form.addParam('inputSignifProts', MultiPointerParam, important=True,
                      label="Input Significant protocols", pointerClass='XmippProtReconstructHeterogeneous',
                      help='Select several protocols of significant heterogeneity '
                           'to create the final setOfVolumes and setOfClasses3D.')

         
    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions -------------------------------

    def createOutputStep(self):

        classListProtocols=[]
        classListIds=[]
        classListSizes=[]

        fnOutFile = self.inputMetaProt.get()._getExtraPath('auxOutputFile.txt')
        outFile = open(fnOutFile, 'r')
        nameRead = outFile.readline()
        while nameRead!='':
            nameRead = nameRead[:-1]
            for i, item in enumerate(self.inputSignifProts):
                if nameRead==self.inputSignifProts[i].get()._objLabel:
                    classListProtocols.append(self.inputSignifProts[i].get())
                    idRead = outFile.readline()
                    classListIds.append(int(idRead))
                    sizeRead = outFile.readline()
                    classListSizes.append(int(sizeRead))
                    break
            nameRead = outFile.readline()
        # print(classListProtocols)
        # print(classListIds)
        # print(classListSizes)

        inputVolume = self.inputMetaProt.get().inputVolume
        myVolList=[]
        # outputVolumes = self._createSetOfVolumes('Def')
        # outputVolumes.setDim(inputVolume.get().getDim())
        # outputVolumes.setSamplingRate(inputVolume.get().getSamplingRate())
        origDim = inputVolume.get().getDim()[0]
        for i, prot in enumerate(classListProtocols):
            # print("createOutputStep ", i)
            classItem = prot.outputClasses[classListIds[i]]
            volFn = str(classItem._representative._filename)
            volFnOut = self._getExtraPath('outVol%d.mrc'%(i+1))
            volDim = classItem._representative.getDim()[0]
            if volDim!=origDim:
                self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d"
                            % (volFn, volFnOut, origDim),
                            numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
            else:
                copy(volFn, volFnOut)
            vol = Volume()
            vol.setLocation(volFnOut)
            vol.setSamplingRate(inputVolume.get().getSamplingRate())
            myVolList.append(vol)
        #     outputVolumes.append(vol)
        # self._defineOutputs(** {'outputVolumesDef': outputVolumes})
        # self._store(outputVolumes)


        inputParticles = self.inputMetaProt.get().inputParticles
        outputDefClasses = self._createSetOfClasses3D(inputParticles.get(), 'Def')
        for i, prot in enumerate(classListProtocols):
            classItem = prot.outputClasses[classListIds[i]]

            signifInputParts = inputParticles.get()
            partIds = classItem.getIdSet()
            newClass = Class3D()
            newClass.copyInfo(signifInputParts)
            newClass.setAcquisition(signifInputParts.getAcquisition())
            newClass.setRepresentative(myVolList[i])

            outputDefClasses.append(newClass)

            enabledClass = outputDefClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in partIds:
                enabledClass.append(signifInputParts[itemId])

            outputDefClasses.update(enabledClass)
        self._defineOutputs(**{'outputClasses3DDef': outputDefClasses})
        self._store(outputDefClasses)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        return summary


