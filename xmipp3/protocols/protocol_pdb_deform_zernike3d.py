
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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

from pwem.protocols import ProtAnalysis3D
from pwem.objects import AtomStruct
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils


class XmippProtDeformPDBZernike3D(ProtAnalysis3D):
    """ Protocol for PDB deformation based on Zernike3D basis. """
    _label = 'deform pdb - Zernike3D'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputPDB', params.PointerParam, label="Input PDB",
                      pointerClass='AtomStruct', important=True,
                      help='Select a PDB to be deformed.')
        form.addParam('inputCoeff', params.PathParam, label="Input Coefficients",
                      important=True,
                      help='Specify a path to the deformation coefficients file.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
        params = ' --pdb %s --clnm %s -o %s' % \
                 (self.inputPDB.get().getFileName(), self.inputCoeff.get(), self._getExtraPath(outFile))
        self.runJob("xmipp_pdb_sph_deform", params)

    def createOutputStep(self):
        outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
        pdb = AtomStruct(self._getExtraPath(outFile))
        self._defineOutputs(outputPDB=pdb)
        self._defineSourceRelation(self.inputPDB, pdb)
