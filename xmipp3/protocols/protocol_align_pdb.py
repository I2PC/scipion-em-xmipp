# **************************************************************************
# *
# * Authors:     David Herreros Calero (ajimenez@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

import sys
import numpy as np
from pwem.convert import AtomicStructHandler
from scipy.spatial import KDTree, ConvexHull

import pyworkflow.protocol.params as params
from pyworkflow.protocol import STEPS_PARALLEL
from pwem.protocols import ProtAnalysis3D
from pwem.objects import AtomStruct, SetOfAtomStructs


class XmippProtAlignPDB(ProtAnalysis3D):
    """
    Aligns a set of atomic structures using ICP (Iterative Closest Point) algorithm
     """
    _label = 'align pdb'

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------

    def _defineParams(self, form):
        form.addSection(label='Input parameters')
        form.addParam('inputReference', params.PointerParam, pointerClass='AtomStruct',
                      label="Reference atomic structure", important=True,
                      help='Reference atomic structure to be used for the alignment.')
        form.addParam('inpuStructures', params.MultiPointerParam, pointerClass='SetOfAtomStructs,AtomStruct',
                      label="Input volume(s)", important=True,
                      help='Select one or more volumes (Volume or SetOfVolumes)\n'
                           'to be aligned againt the reference volume.')

        form.addParallelSection(threads=8)

    # --------------------------- INSERT step functions --------------------------
    def _insertAllSteps(self):
        # Iterate through all input atomic structures and align them
        # againt the reference volume
        refAtoms = self.retrieveAtomCoords(self.inputReference.get().getFileName())
        # refTree = KDTree(refAtoms)

        idx = 1
        alignSteps = []
        for structure in self._iterInputStructures():
            inAtomsFn = structure.getFileName()
            stepId = self._insertFunctionStep('alignAtomsStep', refAtoms, inAtomsFn,
                                              self._getExtraPath("structure%02d.pdb" % idx), prerequisites=[])
            alignSteps.append(stepId)
            idx += 1
        self._insertFunctionStep('createOutputStep', len(alignSteps), prerequisites=alignSteps)

    # --------------------------- STEPS functions --------------------------------------------
    def alignAtomsStep(self, refAtoms_ori, inAtomsFn, outAtomsFn):
        # Retrieve input Atoms
        inAtoms_ori = self.retrieveAtomCoords(inAtomsFn)
        refTree_ori = KDTree(refAtoms_ori)
        center_with_reference = self.createTransformation(np.eye(3), np.mean(refAtoms_ori[:, :3], axis=0))
        rmsd = sys.maxsize

        # Center Atoms in the origin (ICS needs and initial estimation of the alignment)
        for i in range(1):
            if i == 0:  # No Flip
                init_R = np.eye(3)
            elif i == 1:  # Flip X
                init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            elif i == 2:  # Flip Y
                init_R1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                init_R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                init_R = init_R2 @ init_R1
            elif i == 3:  # Flip Z
                init_R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            rotate_input = self.createTransformation(init_R, np.array([0, 0, 0]))
            center_input = self.createTransformation(np.eye(3), -np.mean(inAtoms_ori[:, :3], axis=0))
            center_input = rotate_input @ center_input
            center_ref = self.createTransformation(np.eye(3), -np.mean(refAtoms_ori[:, :3], axis=0))
            inAtoms = np.transpose(np.dot(center_input, inAtoms_ori.T))
            refAtoms = np.transpose(np.dot(center_ref, refAtoms_ori.T))

            ah = AtomicStructHandler()
            ah.read(inAtomsFn)
            ah.writeAsPdb(self._getExtraPath('Input.pdb'))

            ah = AtomicStructHandler()
            ah.read(self.inputReference.get().getFileName())
            ah.writeAsPdb(self._getExtraPath('Ref.pdb'))

            # Compute Convex Hull of atoms and extract boundary coordinates
            boundary_input = ConvexHull(inAtoms[:, :3]).vertices
            boundary_reference = ConvexHull(refAtoms[:, :3]).vertices
            number_input_atoms = len(boundary_input)
            size = int(number_input_atoms * 0.1)
            inAtoms = inAtoms[boundary_input, :]
            refAtoms = refAtoms[boundary_reference, :]
            refTree = KDTree(refAtoms)

            # ICS algorithm with Procrustes rigid aligment
            Tr = np.eye(4)
            for _ in range(10000):

                # Extract initial subset of points and compute closest neighbours
                inAtoms_try = np.transpose(np.dot(Tr, inAtoms.T))
                random_indices = np.random.choice(number_input_atoms, size=size, replace=False)
                # random_indices = np.sort(random_indices)
                points = inAtoms_try[random_indices, :]
                _, index_neighbours = refTree.query(points, k=1)
                neighbours = refAtoms[index_neighbours, :]

                # Removed outliers and compute cost with initial coordinates
                distance_points = np.sqrt(np.sum(np.subtract(points, neighbours)**2, axis=1))
                median_3x = 3 * np.median(distance_points)
                points_cleaned = points[distance_points <= median_3x].reshape(-1, 4)
                neighbours_cleaned = neighbours[distance_points <= median_3x].reshape(-1, 4)
                # cost_start = np.mean(np.linalg.norm(points_cleaned - neighbours_cleaned, axis=1) ** 2)

                # Rigid aligment of subset and new cost
                R_try, t_try = self.rigidRegistration(points_cleaned[:, :3], neighbours_cleaned[:, :3])
                Tr_try = self.createTransformation(R_try, t_try)
                registered_points = np.transpose(np.dot(Tr_try, points_cleaned.T))
                # cost_new = np.mean(np.linalg.norm(registered_points - neighbours_cleaned, axis=1) ** 2)

                # tol_ratio = cost_new / cost_start

                Tr_all = center_with_reference @ Tr_try @ Tr @ center_input
                atoms_aligned = np.transpose(np.dot(Tr_all, inAtoms_ori.T))
                _, index_neighbours = refTree_ori.query(atoms_aligned, k=1)
                neighbours_input = refAtoms_ori[index_neighbours, :]
                rmsd_input_ref = np.sqrt(np.mean(np.linalg.norm(atoms_aligned - neighbours_input, axis=1) ** 2))
                alignedTree = KDTree(atoms_aligned)
                _, index_neighbours = alignedTree.query(refAtoms_ori, k=1)
                neighbours_ref = refAtoms_ori[index_neighbours, :]
                rmsd_ref_input = np.sqrt(np.mean(np.linalg.norm(refAtoms_ori - neighbours_ref, axis=1) ** 2))
                rmsd_trial = 0.5 * (rmsd_input_ref + rmsd_ref_input)

                # We only update the transformation in those interations leading to an improvement
                if (rmsd_trial < rmsd):
                    Tr = np.dot(Tr_try, Tr)
                    rmsd = rmsd_trial
                    # print("Iteration #{} improved aligment by {:2.4%}".format(iteration, 1.0 - tol_ratio))
                    # sys.stdout.flush()

            # Apply Transformations to get final alignment transformation and check final RMSD of alignment
            Tr_final = center_with_reference @ Tr @ center_input
            # atoms_aligned = np.transpose(np.dot(Tr, inAtoms_ori.T))
            # _, index_neighbours = refTree_ori.query(atoms_aligned, k=1)
            # neighbours_aligned = refAtoms_ori[index_neighbours, :]
            # rmsd_trial = np.sqrt(np.mean(np.linalg.norm(atoms_aligned - neighbours_aligned, axis=1) ** 2))
            # print(rmsd_trial)
            # # if rmsd_trial < rmsd:
            # Tr_final = Tr
            # rmsd = rmsd_trial

        print("RMSD of aligned structures is %.2f A" % rmsd)

        # Apply alignment to input structure
        ah = AtomicStructHandler()
        ah.read(inAtomsFn)
        ah.transform(Tr_final)
        ah.writeAsPdb(outAtomsFn)

    def createOutputStep(self, numOutputs):
        output_structures = SetOfAtomStructs(filename=self._getPath('alignedStructs.sqlite'))
        for n in range(numOutputs):
            output_structure = AtomStruct()
            output_structure.setFileName(self._getExtraPath("structure%02d.pdb" % (n+1)))
            output_structures.append(output_structure)
        self._defineOutputs(alignedStucts=output_structures)
        self._defineSourceRelation(self.inputReference, output_structures)


    # --------------------------- UTILS functions --------------------------------------------
    def retrieveAtomCoords(self, structureFile):
        ah = AtomicStructHandler()
        ah.read(structureFile)
        atomIterator = ah.getStructure().get_atoms()
        coordAtoms = []
        for atom in atomIterator:
            coordAtoms.append(np.append(atom.get_coord(), 1))
        return np.asarray(coordAtoms)

    def rigidRegistration(self, X, Y):
        Xcm = np.mean(X, axis=0)
        Ycm = np.mean(Y, axis=0)
        Xc = np.transpose(X - Xcm)
        Yc = np.transpose(Y - Ycm)
        [Ux, _, Vx] = np.linalg.svd(Xc)
        [Uy, _, Vy] = np.linalg.svd(Yc)
        R = Uy @ Ux.T
        t = Ycm - R @ Xcm
        # Xcm = np.sum(X, axis=0) / X.shape[0]
        # Ycm = np.sum(Y, axis=0) / Y.shape[0]
        # Xc = np.transpose(X - Xcm)
        # Yc = np.transpose(Y - Ycm)
        # [U, S, V] = np.linalg.svd(Xc @ Yc.T)
        # R = V @ U.T
        # t = Ycm - R @ Xcm
        return R, t

    def createTransformation(self, R, t):
        tr = np.eye(4)
        tr[:3, :3] = R
        tr[:3, 3] = t
        return tr

    def inverseTransformation(self, Tr):
        R = Tr[:3, :3]
        t = Tr[:3, 3]
        R_T = R.T
        t_inv = np.zeros(3)
        for m in range(3):
            t_inv[m] = -np.dot(R_T[m], t)
        Tr[:3, :3] = R_T
        Tr[:3, 3] = t_inv
        return Tr

    def _iterInputStructures(self):
        """ Iterate over all the input structures. """
        for pointer in self.inpuStructures:
            item = pointer.get()
            if item is None:
                break
            itemId = item.getObjId()
            if isinstance(item, AtomStruct):
                item.outputName = self._getExtraPath('output_strut%06d.pdb' % itemId)
                # If item is a Volume and label is empty
                if not item.getObjLabel():
                    # Volume part of a set
                    if item.getObjParentId() is None:
                        item.setObjLabel("%s.%s" % (pointer.getObjValue(), pointer.getExtended()))
                    else:
                        item.setObjLabel('%s.%s' % (self.getMapper().getParent(item).getRunName(), item.getClassName()))
                yield item
            elif isinstance(item, SetOfAtomStructs):
                for structure in item:
                    structure.outputName = self._getExtraPath('output_structure%06d_%03d.pdb' % (itemId, structure.getObjId()))
                    # If set item label is empty
                    if not structure.getObjLabel():
                        # if set label is not empty use it
                        if item.getObjLabel():
                            structure.setObjLabel("%s - %s%s" % (item.getObjLabel(), structure.getClassName(), structure.getObjId()))
                        else:
                            structure.setObjLabel("%s - %s%s" % (self.getMapper().getParent(item).getRunName(), structure.getClassName(), structure.getObjId()))
                    yield structure
