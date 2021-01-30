
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

import numpy as np

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pwem.emlib.image import ImageHandler
from pwem.objects import AtomStruct
from pyworkflow import VERSION_2_0


class XmippProtPDBDeformSPH(ProtAnalysis3D):
    """ Protocol for pdb deformation based on spherical harmonics. """
    _label = 'sph pdb deform'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('firstPDB', params.PointerParam, label="PDB 1",
                      pointerClass='AtomStruct')
        form.addParam('secondPDB', params.PointerParam, label="PDB 2",
                      pointerClass='AtomStruct')
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed.')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnStruct_1': self._getExtraPath('fnStruct_1.pdb'),
            'fnStruct_2': self._getExtraPath('fnStruct_2.pdb'),
            'fnOutStruct_1': self._getExtraPath('fnStruct_1_deformed.pdb'),
            'fnOutStruct_2': self._getExtraPath('fnStruct_2_deformed.pdb'),
            'outParams': self._getExtraPath('Structures_clnm_plus.txt')
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def convertStep(self):
        pwutils.copyFile(self.firstPDB.get().getFileName(),
                         self._getFileName('fnStruct_1'))
        pwutils.copyFile(self.secondPDB.get().getFileName(),
                         self._getFileName('fnStruct_2'))

    def deformStep(self):
        fnStruct_1 = self._getFileName('fnStruct_1')
        fnStruct_2 = self._getFileName('fnStruct_2')
        outParams = self._getFileName('outParams')

        # self.alignMaps()

        params = ' -a1 %s -a2 %s --analyzeStrain --l1 %d --l2 %d --oroot %s' % \
                 (fnStruct_1, fnStruct_2, self.l1.get(), self.l2.get(),
                  self._getExtraPath('Structures'))
        if self.Rmax != 0:
            params = params + ' --Rmax %d' % self.Rmax
        # if self.numberOfThreads.get() != 0:
        #     params = params + ' --thr %d' % self.numberOfThreads.get()
        self.runJob("xmipp_pseudoatoms_sph_deform", params)

        with open(outParams) as file:
            shift = float(file.readline().split(" ")[-2])
            shift = np.ceil(shift)

        inputPDBs = [fnStruct_1, fnStruct_2]
        for inputPDB in inputPDBs:
            vol = pwutils.removeExt(inputPDB) + "_strain.mrc"
            outPDB = pwutils.removeExt(inputPDB) + "_strain.pdb"
            params = '--pdb %s --vol %s -o %s --sampling 1 --origin %f %f %f --radius 5' % (
                     inputPDB, vol, outPDB, shift, shift, shift)
            self.runJob("xmipp_pdb_label_from_volume", params)
            vol = pwutils.removeExt(inputPDB) + "_rotation.mrc"
            outPDB = pwutils.removeExt(inputPDB) + "_rotation.pdb"
            params = '--pdb %s --vol %s -o %s --sampling 1 --origin %f %f %f --radius 5' % (
                     inputPDB, vol, outPDB, shift, shift, shift)
            self.runJob("xmipp_pdb_label_from_volume", params)

    def createOutputStep(self):
        structure_1 = AtomStruct()
        structure_2 = AtomStruct()
        structure_1.setFileName(self._getFileName('fnOutStruct_1'))
        structure_2.setFileName(self._getFileName('fnOutStruct_2'))
        self._defineOutputs(outputStructure_1=structure_1)
        self._defineSourceRelation(self.firstPDB, structure_1)
        self._defineOutputs(outputStructure_2=structure_2)
        self._defineSourceRelation(self.secondPDB, structure_2)

    # def alignMaps(self):
    #     fnInputVol = self._getFileName('fnInputVol')
    #     fnInputFilt = self._getFileName('fnInputFilt')
    #     fnRefVol = self._getFileName('fnRefVol')
    #     fnRefFilt = self._getFileName('fnRefFilt')
    #     fnOutVol = self._getFileName('fnOutVol')
    #
    #     # Filter the volumes to improve alignment quality
    #     params = " -i %s -o %s --fourier real_gaussian 2" %  (fnInputVol, fnInputFilt)
    #     self.runJob("xmipp_transform_filter", params)
    #     params = " -i %s -o %s --fourier real_gaussian 2" % (fnRefVol, fnRefFilt)
    #     self.runJob("xmipp_transform_filter", params)
    #
    #     # Find transformation needed to align the volumes
    #     params = ' --i1 %s --i2 %s --local --dontScale ' \
    #              '--copyGeo %s' % \
    #              (fnRefFilt, fnInputFilt, self._getExtraPath("geo.txt"))
    #     self.runJob("xmipp_volume_align", params)
    #
    #     # Apply transformation of filtered volume to original volume
    #     with open(self._getExtraPath("geo.txt"), 'r') as file:
    #         geo_str = file.read().replace('\n', ',')
    #     params = " -i %s -o %s --matrix %s" % (fnInputVol, fnOutVol, geo_str)
    #     self.runJob("xmipp_transform_geometry", params)

    # ------------------------- VALIDATE functions -----------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors