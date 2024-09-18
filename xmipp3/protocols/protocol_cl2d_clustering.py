# ******************************************************************************
# *
# * Authors:     Daniel Marchan Torres (da.marchan@cnb.csic.es)
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
# ******************************************************************************

import os.path
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis2D

FN = "class_representatives"

class XmippProtCL2DClustering(ProtAnalysis2D):
    """ 2D clustering protocol to group similar classes """

    _label = '2D classes clustering'

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', params.PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')
        form.addParam('filesPath', params.PathParam,
                      label="File directory",
                      help="Directory to store the 2D representative images.")

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')


    def convertStep(self):
        directoryPath = self.filesPath.get()
        imgsFn = os.path.join(directoryPath, FN + ".mrcs")
        refIdsFn = os.path.join(directoryPath, FN + ".txt")

        if not os.path.exists(directoryPath):
            os.mkdir(directoryPath)

        inputClasses = self.inputClasses.get()

        classes_refIds = []
        for rep in inputClasses.iterRepresentatives():
            idClass, fn = rep.getLocation()
            classes_refIds.append(idClass)

        # Save the corresponding .mrcs file
        inputClasses.writeStack(imgsFn)
        # Save the original ref ids
        with open(refIdsFn, "w") as file:
            for item in classes_refIds:
                file.write(f"{item}\n")


