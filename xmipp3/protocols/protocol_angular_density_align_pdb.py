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

import os, glob
import numpy as np

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, Integer, AtomStruct
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils


class XmippProtAngularDensityAlign(ProtAnalysis3D):
    """ Protocol to rigid align a series of Atomic Strucutres by Angular Density Alignment algorithm. """
    _label = 'pdb angular densitiy align'

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputStructures', params.MultiPointerParam, label="Input Structures",
                      pointerClass='AtomStruct', important=True,
                      help='Select a series of Atomic Structures to be aligned.')
        form.addParam('referenceStructure', params.PointerParam, pointerClass='AtomStruct',
                      label="Reference Structure", important=True,
                      help='Select a reference structure to be used as template for the alignment')
        form.addParam('radius', params.FloatParam, label="Maximum Contact Distance (A)",
                      allowsNull=True,
                      help="If empty, most commonf contact radii will be tested. This will "
                            "help with the convergence of the alignment but execution time will "
                            "increase")
        form.addParam('neighbours', params.IntParam, default=50, label="KDTree Neighbours")
        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self.reference_structure = self.referenceStructure.get()
        for structure_pointer in self.inputStructures:
            structure = structure_pointer.get()
            self._insertFunctionStep("alignStep", structure, prerequisites=[])
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def alignStep(self, structure):
        aligned_structure = self._getExtraPath(pwutils.removeBaseExt(structure.getFileName()) + "_Aligned.pdb")
        params = ' -i %s -r %s --o %s --nn %f' % \
                 (structure.getFileName(), self.reference_structure.getFileName(), aligned_structure,
                  self.neighbours.get())
        if self.radius.get():
            params += " --rmax %f" % self.radius.get()
        self.runJob("xmipp_pdb_angular_density_align", params)

    def createOutputStep(self):
        atom_structure_set = self._createSetOfPDBs()
        for structure_pointer in self.inputStructures:
            structure = structure_pointer.get()
            aligned_file = self._getExtraPath(pwutils.removeBaseExt(structure.getFileName()) + "_Aligned.pdb")
            aligned_structure = AtomStruct(filename=aligned_file)
            atom_structure_set.append(aligned_structure)
        self._defineOutputs(alignedStructures=atom_structure_set)
        self._defineSourceRelation(self.inputStructures, atom_structure_set)
        self._defineSourceRelation(self.referenceStructure, atom_structure_set)

    # ------------------------------- SUMMARY functions -------------------------------
