# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

import numpy as np
import os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO

from pwem.emlib.image import ImageHandler
from pwem.objects import FSC
from pwem.protocols import ProtAnalysis3D
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import STEPS_PARALLEL
import pyworkflow.protocol.params as params
import pwem.emlib.metadata as md

from xmipp3.convert import locationToXmipp, writeSetOfParticles

from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pwem.objects import Volume, SetOfParticles, AtomStruct


FN_METADATA_BFACTOR_RESOLUTION = 'bfactor_resolution.xmd'

class XmippProtbfactorResolution(ProtAnalysis3D):
    """    
    Given a local resolution map and an atomic model, this protocols provides
    the matching between the
    local resolution with the local bfactor per residue.

    AI Generated

    ## Overview

    The Local Resolution/Local B-factor protocol relates a local-resolution map to
    an atomic model.

    Local-resolution maps describe how the estimated resolution varies across a
    cryo-EM reconstruction. Some regions of a map may be well resolved, while
    others may be flexible, poorly ordered, or supported by fewer particles. Atomic
    models also contain per-atom or per-residue B-factor values, which are often
    used to describe local uncertainty, disorder, or mobility.

    This protocol compares these two types of information. It samples the local
    resolution around the atoms of an input atomic model and assigns a
    resolution-derived value to the model. The resulting output is a PDB file in
    which the B-factor column has been replaced by the normalized local-resolution
    information.

    This output can be opened in molecular-visualization tools, such as ChimeraX,
    to color the atomic model according to the local-resolution behavior of the
    map.

    ## Inputs and General Workflow

    The protocol requires:

    - an atomic model;
    - either a local-resolution map or an already normalized local-resolution map;
    - when normalization is requested, the global FSC resolution.

    The input volume is checked and converted to MRC format if needed. The protocol
    then runs the Xmipp local-resolution-to-PDB matching program. For each residue,
    it estimates the local resolution from the surrounding map values, using either
    the median or the mean.

    Finally, it writes an output atomic structure file called `chimeraPDB.pdb`,
    registered in Scipion as the output structure.

    ## Atomic Model

    The **Atomic model** parameter should point to an atomic structure associated
    with the cryo-EM map.

    The atom positions are used to sample the local-resolution map. The protocol
    then estimates a local-resolution value for each residue and writes it into the
    B-factor column of the output PDB file.

    The atomic model and the local-resolution map must be in the same coordinate
    system. If the model is shifted, rotated, or scaled relative to the map, the
    assigned local-resolution values will be incorrect.

    ## Normalize Resolution

    The **Normalize Resolution** option controls whether the protocol normalizes
    the local-resolution map before matching it to the atomic model.

    When this option is enabled, the normalized local-resolution value is computed
    as:

    \[
    \frac{LR - FSC}{FSC}
    \]

    where \(LR\) is the local resolution at a voxel and \(FSC\) is the global FSC
    resolution in angstroms.

    This normalized value indicates whether a region is locally better or worse
    than the global FSC resolution. Values above zero indicate local resolution
    worse than the global reference; values below zero indicate local resolution
    better than the global reference.

    When this option is disabled, the protocol assumes that the input map is
    already a normalized local-resolution map.

    ## Local Resolution Map

    The **Local Resolution Map** parameter is used when normalization is enabled.

    This map should contain local-resolution values in angstroms. The user must
    also provide the global **FSC resolution** so that the protocol can compute the
    normalized value.

    The local-resolution map should correspond to the same reconstruction and
    coordinate frame as the atomic model.

    ## Normalized Local Resolution Map

    The **Normalized Local Resolution Map** parameter is used when normalization is
    disabled.

    In this case, the input map is assumed to already contain values of the form:

    \[
    \frac{LR - FSC}{FSC}
    \]

    The protocol does not normalize the map again. It directly uses the provided
    values for matching to the atomic model.

    This option is useful when the normalized map has already been computed by a
    previous workflow.

    ## FSC Resolution

    The **FSC resolution** parameter is required when the protocol normalizes the
    local-resolution map.

    This value is the global resolution of the map in angstroms, usually obtained
    from an FSC criterion such as FSC = 0.143 or FSC = 0.5, depending on the
    workflow.

    The choice of FSC resolution affects the normalized values. Therefore, users
    should report which FSC criterion was used when interpreting or presenting the
    result.

    ## Use Median

    The **Use median** option controls how the local-resolution value is estimated
    for each residue.

    If enabled, the protocol uses the median of local-resolution values around the
    residue. The median is more robust to outliers and is usually a good choice
    when the local-resolution map contains noisy or extreme voxels.

    If disabled, the protocol uses the mean. The mean may be more sensitive to all
    values in the neighborhood, but it can also be more affected by outliers.

    For most biological interpretation, the median is a sensible default.

    ## Is the Atomic Model Centered?

    The **is the atomic centered** option tells the protocol whether the atomic
    model is centered in the middle of the local-resolution map.

    This affects how atom coordinates are interpreted relative to the volume grid.
    If the model is centered as expected, the option should be enabled.

    If the coordinate convention differs, the user should disable this option or
    ensure that the model and map are properly aligned before running the protocol.

    Incorrect centering can produce wrong residue-level assignments.

    ## Output Structure

    The main output is **outputStructure**.

    This is an atomic structure file named `chimeraPDB.pdb`. In this output, the
    B-factor column is replaced by the normalized local-resolution value assigned
    to each atom or residue.

    The output can be opened in molecular viewers and colored by B-factor. In that
    case, the coloring will reflect local-resolution-derived values rather than
    the original crystallographic or model B-factors.

    This output is mainly intended for visualization and interpretation.

    ## Interpreting the Output

    The output structure should be interpreted as a visualization bridge between
    map quality and atomic-model location.

    Regions with values above zero correspond to areas where the local resolution
    is worse than the global FSC resolution. Regions with values below zero
    correspond to areas where the local resolution is better than the global FSC
    resolution.

    The values are not the original atomic B-factors. They are local-resolution
    annotations stored in the B-factor column for convenience.

    This distinction is important when using downstream tools that display or
    analyze B-factor values.

    ## Practical Recommendations

    Use an atomic model that is correctly fitted into the map before running this
    protocol.

    Make sure that the local-resolution map and the atomic model are in the same
    coordinate frame.

    Use normalization when the input map contains local-resolution values in
    angstroms. Disable normalization only if the input map is already normalized.

    Choose the global FSC resolution carefully and record the criterion used to
    obtain it.

    Use the median option for robust residue-level estimates, especially when the
    local-resolution map is noisy.

    After the protocol finishes, open the output PDB in ChimeraX or another viewer
    and color by B-factor to inspect the spatial distribution of local-resolution
    quality.

    Remember that the output B-factor column no longer contains conventional
    atomic B-factors.

    ## Final Perspective

    Local Resolution/Local B-factor is a visualization and interpretation protocol
    that maps local-resolution information onto an atomic model.

    For biological users, its main value is that it makes map-quality variation
    visible directly on the molecular structure. This helps identify which domains,
    loops, interfaces, or residues are supported by better or worse local map
    resolution.

    The protocol should be used with properly aligned maps and models, and the
    result should be interpreted as local-resolution annotation rather than as a
    new atomic B-factor refinement.
    """
    _label = 'local resolution/local bfactor'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        self.vol = ''
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('pdbfile', PointerParam, pointerClass='AtomStruct',
                      label="Atomic model", important=True,
                      help='Select an atomic model. The atom positions will be taken'
                           ' to estimate the local resolution around them and then, the '
                           ' local resolution associated to each residue.')

        form.addParam('normalizeResolution', BooleanParam, default=True,
                      label="Normalize Resolution",
                      help='The normalizedlocal resolution map is defined as '
                           '(LR - FSC)/FSC, where LR is the local resolution of a '
                           'given voxel, and FSC is the FSC resolution in A. This '
                           'map provides information about whether the local resolution is '
                           'greater or lesser than the FSC. The local resolution '
                           'normalized map is used to carry out the matching with the local '
                           'bfactor per residue. Yes means that the local resolution will be '
                           'normalized by the algorithm. No means that the input local '
                           'resolution map is already a normalized local resolution map.')

        form.addParam('localResolutionMap', PointerParam, pointerClass='Volume',
                      label="Local Resolution Map", important=True,
                      condition='normalizeResolution',
                      help='Select a local resolution map. Alternatively, the input.'
                           ' can be a normalized local resolution map, in this case'
                           ' set the Normalize resolution to No')

        form.addParam('normalizedMap', PointerParam, pointerClass='Volume',
                      label="Normalized Local Resolution Map", important=True,
                      condition='not normalizeResolution',
                      help='Select a normalized local resolution map. The local'
                           ' resolution normalized map is defined as '
                           '(LR - FSC)/FSC, where LR is the local resolution of a'
                           ' given voxel, and FSC is the FSC resolution in A')

        form.addParam('fscResolution', FloatParam,
                      condition = 'normalizeResolution',
                      label="FSC resolution (A)",
                      help='The global resolution of the map in A')

        form.addParam('medianEstimation', BooleanParam, default=True,
                      label="Use median",
                      help='The local resolution per residue can be estimated using '
                           'the mean (by default - No) or the median (yes)')

        form.addParam('centered', BooleanParam, default=True,
                      label="is the atomic centered",
                      help='True if the atomic model centered in midle of'
                           ' the local resolution map')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):

        # 1 Check the input and convert to mrc if it is the case
        self._insertFunctionStep(self.convertInputStep)

        # 2 Carry out the matching betweent the local resolution per residue and bfactor
        self._insertFunctionStep(self.matchingBfactorLocalResolution)

        self._insertFunctionStep(self.createOutputStep)

    def mrc_convert(self, fileName, outputFileName):
        """Check if the extension is .mrc, if not then uses xmipp to convert it
        """
        ext = getExt(fileName)
        if ((ext != '.mrc') and (ext != '.map')):
            params = ' -i %s' % fileName
            params += ' -o %s' % outputFileName
            self.runJob('xmipp_image_convert', params)
            return outputFileName
        else:
            return fileName

    def convertInputStep(self):
        """ Read the input volume and check the file extension to convert to mrc is it is the case.
        """
        if self.normalizeResolution.get():
            self.inputMap = self.localResolutionMap
        else:
            self.inputMap = self.normalizedMap

        self.vol = self.mrc_convert(self.inputMap.get().getFileName(),
                                    self._getTmpPath('localResolutionMap.mrc'))        

    def matchingBfactorLocalResolution(self):
        """ The local resolution map and the pdb are taken and analyzed to match the
        local resolution and bfactor per residue. The output will be a pdb file with the
        bfactor column substituted by the normalized local resolution. This is the (local
        resolution - fscResolution)/fscResolution.
        """
        fnPDB = self.pdbfile.get().getFileName()
        fnList, chainList = self.splitPdbByChains(fnPDB, output_folder=self._getExtraPath())
        print(fnList)
        for idx, fn in enumerate(fnList):
            params = ' --atmodel %s' % fn
            params += ' --vol %s' % self.vol
            if self.normalizeResolution.get():
                params += ' --fscResolution %s' % self.fscResolution.get()
            params += ' --sampling %f' % self.inputMap.get().getSamplingRate()
            if self.medianEstimation.get():
                params += ' --useMedian '
            if self.centered.get():
                params += ' --centered '
            params += ' -o %s' % self._getExtraPath()

            self.runJob('xmipp_resolution_pdb_bfactor', params)
            chain = str(chainList[idx])
            src = self._getExtraPath('bfactor_resolution.xmd')
            dst = self._getExtraPath('bfactor_resolution'+chain+'.xmd')

            os.rename(src, dst)

            src = self._getExtraPath('chimeraPDB.pdb')
            dst = self._getExtraPath('chimeraPDB_'+chain+'.pdb')

            os.rename(src, dst)

        pdbFull = self.convertCifToPDB(fnPDB)
        params = ' --atmodel %s' % pdbFull
        params += ' --vol %s' % self.vol
        if self.normalizeResolution.get():
            params += ' --fscResolution %s' % self.fscResolution.get()
        params += ' --sampling %f' % self.inputMap.get().getSamplingRate()
        if self.medianEstimation.get():
            params += ' --useMedian '
        if self.centered.get():
            params += ' --centered '
        params += ' -o %s' % self._getExtraPath()

        self.runJob('xmipp_resolution_pdb_bfactor', params)


    def createOutputStep(self):
        outputPdb = AtomStruct()
        outputPdb.setFileName(self._getExtraPath('chimeraPDB.pdb'))
        self._defineOutputs(outputStructure=outputPdb)
    
    def convertCifToPDB(self, file_path):
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
    
        if ext.lower() == ".pdb":
            parser = PDBParser(QUIET=True)
        elif ext.lower() == ".cif":
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Wrong extension it is not a .pdb or .cif')

        structure = parser.get_structure(base_name, file_path)
        io = PDBIO()
        io.set_structure(structure)

        pdbFull = self._getExtraPath(base_name+'.pdb')
        io.save(pdbFull)
        return pdbFull

    def splitPdbByChains(self, file_path, output_folder='splitChains'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
    
        if ext.lower() == ".pdb":
            parser = PDBParser(QUIET=True)
        elif ext.lower() == ".cif":
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Wrong extension it is not a .pdb or .cif')

        structure = parser.get_structure(base_name, file_path)

        io = PDBIO()

        model = structure[0]
        fnList = []
        chainList = []

        for chain in model:
            chain_id = chain.get_id()
        
            out_name = f'{base_name}chain{chain_id}.pdb'
            out_path = os.path.join(output_folder, out_name)

            io.set_structure(chain)
            io.save(out_path)
        
            fnList.append(out_path)
            chainList.append(chain_id)
            self.parseAndFixPDB(out_path)
        return fnList, chainList

    def parseAndFixPDB(self, fnPDB):
        with open(fnPDB, 'r+') as pdbFile:
            allLines = pdbFile.readlines()
            pdbFile.seek(0)
            for line in allLines:
                bfactorStr = line[60:66]
                try:
                    bfactorFloat = float(bfactorStr)
                    fixedBfactor = str(bfactorFloat)[:5]

                    line2 = line[0:60] + ' ' + fixedBfactor + line[67::]# + '\n'

                    pdbFile.write(line2)
                except:
                    continue

        pdbFile.close()
