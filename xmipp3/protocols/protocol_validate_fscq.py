# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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

import os, shutil, gzip
from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, BooleanParam,
                                        IntParam, FileParam, FloatParam)
import pwem.emlib.metadata as md
from pwem.emlib.metadata import (MDL_VOLUME_SCORE1, MDL_VOLUME_SCORE2)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils import getExt
from pyworkflow.object import (Float, Integer)
from pwem.objects import AtomStruct
from pwem.convert import Ccp4Header
from pwem.convert.atom_struct import toPdb, toCIF, AtomicStructHandler, addScipionAttribute
from pwem.emlib import Image
from pyworkflow import BETA, UPDATED, NEW, PROD


VALIDATE_METHOD_URL = 'https://github.com/I2PC/scipion-em-xmipp/wiki/XmippProtValFit'
OUTPUT_PDBVOL_FILE = 'pdbVol'
OUTPUT_PDBMRC_FILE = 'pdb_volume.map'
BLOCRES_AVG_FILE = 'blocresAvg'
BLOCRES_HALF_FILE = 'blocresHalf'
RESTA_FILE = 'diferencia.vol'
RESTA_FILE_MRC = 'diferencia.map'
MASK_FILE_MRC = 'mask.map'
MASK_FILE = 'mask.vol'
FN_VOL = 'vol.map'
FN_HALF1 = 'half1.map'
FN_HALF2 = 'half2.map'
MD_MEANS = 'params.xmd'
MD2_MEANS = 'params2.xmd'
RESTA_FILE_NORM = 'diferencia_norm.map'
OUTPUT_CIF = 'fscq_struct.cif'


class XmippProtValFit(ProtAnalysis3D):
    """
   Assesses the quality of the fit between a model and experimental data. This
   protocol evaluates how well a volume or structure matches reference data,
   guiding improvements in model accuracy.

   AI Generated

    ## Overview

    The Validate FSC-Q protocol evaluates the local agreement between an atomic
    model and a cryo-EM map, using the information contained in the two half maps.

    The protocol compares two kinds of local resolution or local agreement:

    - the agreement between the experimental map and a map generated from the
      atomic model;
    - the agreement between the two experimental half maps.

    The difference between these quantities is used to estimate how much the atomic
    model deviates from the signal supported by the experimental data. This
    information is then assigned to the atoms of the input structure as an FSC-Q
    score.

    The main output is an atomic structure in CIF format containing an additional
    Scipion atom-level attribute named `fscq_score`. This structure can be used to
    visualize which parts of the model agree better or worse with the experimental
    map.

    ## Inputs and General Workflow

    The protocol requires:

    - an input volume with associated half maps;
    - an atomic model;
    - optionally, a volume generated from the atomic model;
    - optionally, a soft mask.

    The protocol first converts or links the atomic model to a suitable structure
    file. It prepares the experimental map and the two half maps with the correct
    sampling rate and origin. If no map from the atomic model is provided, the
    protocol creates one automatically from the atomic coordinates and locally
    aligns it to the experimental map.

    The protocol then computes local resolution maps using Bsoft `blocres`:

    - one local map comparing the experimental map and the PDB-derived map;
    - one local map comparing half map 1 and half map 2.

    It subtracts the half-map agreement from the map-model agreement, producing an
    FSC-Q-like difference map. It also creates a normalized version divided by the
    local resolution. Finally, it samples these maps around the atoms of the
    atomic model and writes the resulting FSC-Q values into output structure files.

    ## Input Volume

    The **Input Volume** parameter defines the experimental cryo-EM map used for
    validation.

    This volume must have associated half maps. The protocol validates this
    requirement. If the volume was imported into Scipion, the half maps must also
    have been imported and linked to the volume.

    The sampling rate of this volume is used throughout the calculation. The input
    volume, half maps, mask, PDB-derived map, and atomic model must all correspond
    to the same structure and coordinate frame.

    ## Atomic Model Input

    The atomic model can be provided in two ways.

    If **Input PDB from file** is disabled, the user selects a Scipion
    **AtomStruct** object through the **Refined PDB** parameter.

    If **Input PDB from file** is enabled, the user provides a file path through
    the **PDB File path** parameter.

    The protocol accepts common atomic-structure formats such as PDB, ENT, CIF,
    mmCIF, and compressed CIF files. Internally, it prepares the structure for
    both PDB-style processing and CIF output annotation.

    The atomic model should already be fitted into the input map. A poorly fitted
    model will produce misleading FSC-Q values.

    ## Volume from PDB

    The **Volume from PDB** parameter allows the user to provide a map already
    generated from the atomic model.

    This volume should be aligned with the experimental reconstruction. Providing
    it can save time and gives the user control over how the model-derived density
    was generated.

    If no PDB-derived volume is provided, the protocol creates one automatically
    from the atomic model using the input map sampling rate and box size. It then
    locally aligns the generated volume to the experimental map.

    The PDB-derived volume is used to compare the atomic model with the
    experimental density.

    ## Soft Mask

    The **Soft Mask** parameter defines the region used for the local-resolution
    and map-model comparison.

    If a mask is provided, it is converted to the working map format with the
    correct origin and sampling rate.

    If no mask is provided, the protocol creates an automatic mask from the
    PDB-derived map. The automatic mask is obtained by thresholding the model map
    and applying a dilation operation.

    The mask should include the molecular region to be evaluated while excluding
    unnecessary background. A poor mask can affect local agreement estimates.

    ## Window Size

    The **window size** parameter defines the size of the sliding local window used
    for local resolution estimation.

    The value is expressed in pixels or voxels. The default is 20.

    A smaller window gives more localized estimates but can be noisier. A larger
    window gives smoother and more stable estimates but may blur local differences
    between nearby regions.

    The chosen value should be appropriate for the map resolution, box size, and
    level of spatial detail desired in the validation.

    ## Set Origin of Coordinates

    The **Set origin of coordinates** option allows the user to define an explicit
    origin offset for the maps.

    When enabled, the user provides X, Y, and Z coordinate offsets. The protocol
    uses these values to create map files with the corresponding origin.

    This option is useful when the atomic model and map require a specific origin
    convention to be aligned correctly.

    If disabled, the protocol uses zero offsets.

    Correct origin handling is essential. If the model and map are shifted relative
    to each other, FSC-Q values will not represent true local agreement.

    ## Half-Map Local Resolution

    One branch of the protocol computes a local agreement map between the two half
    maps.

    This calculation estimates the local signal supported reproducibly by the two
    independent reconstructions. It uses Bsoft `blocres` with a cutoff of 0.5.

    This half-map local agreement is the experimental baseline against which the
    map-model agreement is compared.

    ## Map-Model Local Resolution

    The other branch computes local agreement between the experimental map and the
    PDB-derived map.

    This calculation uses Bsoft `blocres` with a cutoff of 0.67. It estimates how
    well the atomic model-derived density agrees locally with the experimental map.

    The comparison is performed inside the selected or automatically generated
    mask.

    ## FSC-Q Difference Map

    After both local maps have been computed, the protocol subtracts the half-map
    local result from the map-model local result.

    The resulting difference map is written as `diferencia.map`.

    Conceptually, this map represents how much the model-map agreement deviates
    from the agreement supported by the experimental half maps. Regions where the
    model behaves differently from the half-map signal may indicate local model
    problems, poor fit, flexibility, or map/model mismatch.

    ## Normalized FSC-Qr Map

    The protocol also creates a normalized difference map, written as
    `diferencia_norm.map`.

    This map divides the FSC-Q difference by the local half-map resolution wherever
    the mask and local-resolution values are valid.

    The normalized value, referred to in the summary as FSC-Qr, provides an
    alternative score scaled by local resolution. This can help compare deviations
    across regions with different local resolution.

    ## Assigning FSC-Q to Atoms

    The protocol samples the FSC-Q difference maps around the atoms of the atomic
    model.

    It uses a sampling radius of 0.8 and writes the sampled values into
    structure files. The FSC-Q value is first stored in the occupancy field of
    intermediate structure files, and then transferred into a Scipion CIF
    attribute named `fscq_score`.

    This produces an atom-level validation annotation that can be visualized or
    analyzed later.

    ## Output Atomic Structure

    The main output is **outputAtomStruct**.

    This output is a CIF file named `fscq_struct.cif`. It contains the input atomic
    model with an additional atom-level Scipion attribute:

    `fscq_score`

    The output structure is linked to the input cryo-EM volume.

    If the input atomic model was provided as a Scipion AtomStruct object, the
    protocol also defines the corresponding transform relation.

    ## Summary Metrics

    The protocol summary reports several global FSC-Q statistics.

    For the raw FSC-Q score, it reports:

    - mean FSC-Q;
    - absolute mean FSC-Q.

    For the normalized FSC-Qr score, it reports:

    - mean FSC-Qr;
    - absolute mean FSC-Qr.

    It also reports:

    - total number of atoms analyzed;
    - number and percentage of atoms with FSC-Q greater than 0.5;
    - number and percentage of atoms with FSC-Q less than -0.5.

    These values provide a compact summary of the map-model deviations across the
    atomic model.

    ## Interpreting FSC-Q Values

    FSC-Q values should be interpreted as local map-model agreement annotations.

    Regions with large positive or negative deviations may indicate parts of the
    model whose agreement with the map differs from the local experimental
    half-map signal. Such regions deserve closer inspection.

    Potential causes include poor local fitting, wrong side-chain placement,
    incorrect domain position, flexibility, low occupancy, local map weakness,
    masking effects, or errors in coordinate origin or alignment.

    The score is not a substitute for visual inspection. It should be used together
    with the density map, half maps, local resolution, model geometry validation,
    and biological knowledge.

    ## Requirements and Validation

    The input volume must have associated half maps. If the half maps are missing,
    the protocol reports an error.

    The user must provide an atomic model, either as a Scipion AtomStruct object or
    as a file path.

    The protocol also requires the Bsoft plugin, version 3.0.5 or higher. If the
    plugin is missing or an older unsupported version is detected, validation
    reports an error.

    ## Practical Recommendations

    Use this protocol after the atomic model has been fitted and refined against
    the cryo-EM map.

    Make sure the input volume has correctly associated half maps.

    Check that the atomic model, experimental map, half maps, PDB-derived map, and
    mask are all in the same coordinate frame.

    Provide a PDB-derived map if you have already generated one with the desired
    parameters and alignment. Otherwise, allow the protocol to generate it
    automatically.

    Use a mask that covers the molecular region without including excessive
    background.

    Use the default window size as a starting point, then adjust it if the local
    scores appear too noisy or too smoothed.

    Inspect regions with large FSC-Q deviations in a molecular viewer together
    with the original map and half maps.

    ## Final Perspective

    Validate FSC-Q is a map-model validation protocol.

    For biological users, its main value is that it projects local map-model
    agreement onto the atomic model, producing atom-level FSC-Q annotations. This
    helps identify which residues or atomic regions agree well with the
    experimental density and which regions may need closer inspection.

    The protocol is most useful as part of a broader validation workflow, together
    with visual map inspection, half-map FSC, local resolution, model geometry
    validation, and biological interpretation.
    """
    _label = 'validate fsc-q'
    _lastUpdateVersion = VERSION_3_0
    _devStatus = PROD
    _ATTRNAME = 'fscq_score'
    _OUTNAME = 'outputAtomStruct'
    _possibleOutputs = {_OUTNAME: AtomStruct}

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = 1

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        group = form.addGroup('Input')
        group.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      help='Select a volume.')

        group.addParam('fromFile', BooleanParam, default=False,
                      label='Input PDB from file: ')
        group.addParam('inputPDBObj', PointerParam, pointerClass='AtomStruct', allowsNull=True,
                      label="Refined PDB: ", important=True, condition='not fromFile',
                      help='Specify the desired input structure.')
        group.addParam('inputPDB', FileParam,  condition='fromFile',
                      label="PDB File path: ", important=True,
                      help='Specify a path to desired PDB structure.')

        group.addParam('pdbMap_', PointerParam, pointerClass='Volume',
                      label="Volume from PDB: ", allowsNull=True,
                      help='Volume created from the PDB.'
                           ' The volume should be aligned with the reconstruction map.'
                           ' If the volume is not entered,' 
                           ' it is automatically created from the PDB.')

        group.addParam('inputMask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Soft Mask",
                      help='The mask determines which points are specimen'
                      ' and which are not. If the mask is not passed,' 
                      ' the method creates an automatic mask from the PDB.')

        group = form.addGroup('Parameters')
        group.addParam('box', IntParam, default=20,
                      label="window size",
                      help='Kernel size (slidding window) for determining'
                      ' local resolution (pixels/voxels).')

        group.addParam('setOrigCoord', BooleanParam,
                      label="Set origin of coordinates",
                      help="Option YES:\nA new volume will be created with "
                           "the given ORIGIN of coordinates. ",
                      default=False)

        group.addParam('xcoor', FloatParam, default=0, condition='setOrigCoord',
                      label="x", help="offset along x axis")
        group.addParam('ycoor', FloatParam, default=0, condition='setOrigCoord',
                      label="y", help="offset along y axis")
        group.addParam('zcoor', FloatParam, default=0, condition='setOrigCoord',
                      label="z", help="offset along z axis")

        form.addParallelSection(threads=8, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
                 OUTPUT_PDBVOL_FILE: self._getTmpPath('pdb_volume'),
                 OUTPUT_PDBMRC_FILE: self._getExtraPath('pdb_volume.map'),
                 BLOCRES_AVG_FILE: self._getTmpPath('blocres_avg.map'),
                 BLOCRES_HALF_FILE: self._getTmpPath('blocres_half.map'),
                 RESTA_FILE: self._getTmpPath('diferencia.vol'),
                 RESTA_FILE_MRC: self._getExtraPath('diferencia.map'),
                 RESTA_FILE_NORM: self._getExtraPath('diferencia_norm.map'),
                 MASK_FILE_MRC : self._getExtraPath('mask.map'),
                 MASK_FILE: self._getTmpPath('mask.vol'),
                 FN_VOL: self._getTmpPath("vol.map"),
                 FN_HALF1: self._getTmpPath("half1.map"),
                 FN_HALF2: self._getTmpPath("half2.map"),
                 MD_MEANS: self._getExtraPath('params.xmd'),
                 MD2_MEANS: self._getExtraPath('params2.xmd')
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):

        self._createFilenameTemplates()
        input = self._insertFunctionStep('convertInputStep')
        id = []
        for i in range(2):
            id.append(self._insertFunctionStep('runBlocresStep', i, prerequisites=[input]))
        input1 = self._insertFunctionStep('substractBlocresStep',prerequisites=id)
        input2 = self._insertFunctionStep('assignPdbStep', prerequisites=[input1])
        self._insertFunctionStep('createOutputStep', prerequisites=[input2])


    def convertInputStep(self):
        """ Convert inputs to desired format."""
        #Convert Input to pdb
        if self.isStructExtensionValid():
            os.symlink(os.path.abspath(self.getInputStructFile()), self.getStructFile())
        else:
            toPdb(self.getInputStructFile(), self.getStructFile())

        """ Read the input volume."""
        self.volume = self.inputVolume.get()

        """Read the Origin."""
        if self.setOrigCoord.get():
            self.x = self.xcoor.get()
            self.y = self.ycoor.get()
            self.z = self.zcoor.get()
        else:
            self.x = 0
            self.y = 0
            self.z = 0

        self.sampling = self.volume.getSamplingRate()
        self.origin = (-self.x*self.sampling, -self.y*self.sampling, -self.z*self.sampling)

        self.vol = self.volume.getFileName()
        self.half1, self.half2 = self.volume.getHalfMaps().split(',')

        extVol = getExt(self.vol)
        if (extVol == '.mrc') or (extVol == '.map'):
            self.vol_xmipp = self.vol + ':mrc'
        else:
            self.vol_xmipp = self.vol

        self.fnvol = self._getFileName(FN_VOL)
        self.fnvol1 = self._getFileName(FN_HALF1)
        self.fnvol2 = self._getFileName(FN_HALF2)

        Ccp4Header.fixFile(self.vol, self.fnvol, self.origin, self.sampling,
                        Ccp4Header.START)
        Ccp4Header.fixFile(self.half1, self.fnvol1, self.origin, self.sampling,
                        Ccp4Header.START)
        Ccp4Header.fixFile(self.half2, self.fnvol2, self.origin, self.sampling,
                        Ccp4Header.START)

        """Create map from PDB """
        if self.pdbMap_.hasValue():
            pdbvolume = self.pdbMap_.get()
            self.pdbvol = pdbvolume.getFileName()

            self.pdbmap = self._getFileName(OUTPUT_PDBMRC_FILE)
            Ccp4Header.fixFile(self.pdbvol, self.pdbmap, self.origin, self.sampling,
                        Ccp4Header.START)

        else:
            """ Convert PDB to Map """
            params = ' --centerPDB '
            params += ' -v 0 '
            params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
            params += ' --size %d' % self.inputVolume.get().getXDim()
            params += ' -i %s' % self.getStructFile()
            params += ' -o %s' % self._getFileName(OUTPUT_PDBVOL_FILE)
            self.runJob('xmipp_volume_from_pdb', params)

            """ Align pdbMap to reconstruction Map """

            params = ' --i1 %s' % self.vol_xmipp
            params += ' --i2 %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
            params += ' --local --apply'
            params += ' %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
            self.runJob('xmipp_volume_align', params)

            """ convert align vol to mrc format """

            self.pdbvol = self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
            self.pdbmap = self._getFileName(OUTPUT_PDBMRC_FILE)
            Ccp4Header.fixFile(self.pdbvol, self.pdbmap, self.origin, self.sampling,
                        Ccp4Header.START)


        """ Create a mask"""
        if self.inputMask.hasValue():
            self.maskIn = self.inputMask.get().getFileName()
            self.maskFn = self._getFileName(MASK_FILE_MRC)
            Ccp4Header.fixFile(self.maskIn, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)

            self.mask_xmipp = self.maskFn + ':mrc'

        else:

            self.maskFn = self._getFileName(MASK_FILE_MRC)
            self.mask_xmipp = self._getFileName(MASK_FILE)

            if (not self.pdbMap_.hasValue()):

                params = ' -i %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
                params += ' -o %s' % self.mask_xmipp
                params += ' --select below 0.02 --substitute binarize'
                self.runJob('xmipp_transform_threshold', params)

                params = ' -i %s' % self.mask_xmipp
                params += ' -o %s' % self.mask_xmipp
                params += ' --binaryOperation dilation --size 3'
                self.runJob('xmipp_transform_morphology', params)

                """ convert mask.vol to mrc format """
                self.maskFn = self._getFileName(MASK_FILE_MRC)
                Ccp4Header.fixFile(self.mask_xmipp, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)

            else:

                """ create mask from pdbMap """
                params = ' -i %s' % self._getFileName(OUTPUT_PDBMRC_FILE)+':mrc'
                params += ' -o %s' % self.mask_xmipp
                params += ' --select below 0.02 --substitute binarize'
                self.runJob('xmipp_transform_threshold', params)

                params = ' -i %s' % self.mask_xmipp
                params += ' -o %s' % self.mask_xmipp
                params += ' --binaryOperation dilation --size 3'
                self.runJob('xmipp_transform_morphology', params)

                """ convert mask.vol to mrc format """
                self.maskFn = self._getFileName(MASK_FILE_MRC)
                Ccp4Header.fixFile(self.mask_xmipp, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)


    def runBlocresStep(self, i):
        # Local import to prevent discovery errors
        import bsoft

        if (i==0):

            """ Calculate FSC map-PDB """

            params = '  -nofill -smooth -pad 1 '
            params += ' -cutoff 0.67'
            params += ' -maxresolution 0.5 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate())
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2]))
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol, self._getFileName(OUTPUT_PDBMRC_FILE))
            params += ' %s' % self._getFileName(BLOCRES_AVG_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres'), params)
        else:

            """ Calculate FSC half1-half2 """

            params = '  -nofill -smooth -pad 1 '
            params += ' -cutoff 0.5'
            params += ' -maxresolution 0.5 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate())
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2]))
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol1, self.fnvol2)
            params += ' %s' % self._getFileName(BLOCRES_HALF_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres'), params)

    def substractBlocresStep(self):

        params = ' -i %s' % self._getFileName(BLOCRES_AVG_FILE)+':mrc'
        params += ' --minus %s' % self._getFileName(BLOCRES_HALF_FILE)+':mrc'
        params += ' -o %s ' % self._getFileName(RESTA_FILE)
        self.runJob('xmipp_image_operate', params)

        Ccp4Header.fixFile(self._getFileName(RESTA_FILE), self._getFileName(RESTA_FILE_MRC),
                           self.origin, self.sampling, Ccp4Header.START)

        """Diveded by resolution"""
        Vx = Image(self._getFileName(RESTA_FILE))
        V=Vx.getData()
        Vmask = Image(self._getFileName(MASK_FILE_MRC)+':mrc').getData()
        Vres = Image(self._getFileName(BLOCRES_HALF_FILE)+':mrc').getData()
        Vt = V
        Zdim, Ydim, Xdim = V.shape

        for z in range(0,Zdim):
            for y in range(0,Ydim):
                for x in range(0,Xdim):
                    if (Vmask[z,y,x] > 0.001 and Vres[z,y,x]>0.001):
                        Vt[z,y,x] = (V[z,y,x]/Vres[z,y,x])
        Vx.setData(Vt)
        Vx.write(self._getFileName(RESTA_FILE_NORM))
        Ccp4Header.fixFile(self._getFileName(RESTA_FILE_NORM), self._getFileName(RESTA_FILE_NORM),
                           self.origin, self.sampling, Ccp4Header.START)


    def assignPdbStep(self):

        params = ' --pdb %s ' % self.getStructFile()
        params += ' --vol %s ' % self._getFileName(RESTA_FILE_MRC)
        params += ' --mask %s ' % self.mask_xmipp
        params += ' -o %s ' % self.getFSCQFile()
        params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
        params += ' --origin %f %f %f' %(self.x, self.y, self.z)
        params += ' --radius 0.8'
        params += ' --md %s' % self._getFileName(MD_MEANS)
        self.runJob('xmipp_pdb_label_from_volume', params)

        """Diveded by resolution"""
        params = ' --pdb %s ' % self.getStructFile()
        params += ' --vol %s ' % self._getFileName(RESTA_FILE_NORM)
        params += ' --mask %s ' % self.mask_xmipp
        params += ' -o %s ' % self.getNormFSCQFile()
        params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
        params += ' --origin %f %f %f' %(self.x, self.y, self.z)
        params += ' --radius 0.8'
        params += ' --md %s' % self._getFileName(MD2_MEANS)
        self.runJob('xmipp_pdb_label_from_volume', params)

    def _setMetrics(self):
        """ Internal method to compute some metrics. """
        # mean values of FSC-Q

        mtd = md.MetaData()
        mtd.read(self._getFileName(MD_MEANS))

        mean = mtd.getValue(MDL_VOLUME_SCORE1, 1)
        meanA = mtd.getValue(MDL_VOLUME_SCORE2, 1)

        # means value for map divided by resolution (FSC-Qr)
        mtd2 = md.MetaData()
        mtd2.read(self._getFileName(MD2_MEANS))

        mean2 = mtd2.getValue(MDL_VOLUME_SCORE1, 1)
        meanA2 = mtd2.getValue(MDL_VOLUME_SCORE2, 1)

        # statistic from fnal pdb with fsc-q
        # Number of atoms greater or less than 0.5
        totalAtom = 0
        fscqGreater = 0
        fscqLess = 0

        # Reading FCSQ's value file
        atStHandler = AtomicStructHandler()
        atStHandler.read(self.getFSCQFile())

        # Reading value stored in occupancy field for each atom
        for model in atStHandler.structure:
            for atom in model.get_atoms():
                totalAtom += 1
                fscqAtom = atom.get_occupancy()

                if (fscqAtom > 0.5):
                    fscqGreater += 1

                if (fscqAtom < -0.5):
                    fscqLess += 1

        porcGreater = (fscqGreater * 100) / totalAtom
        porcLess = (fscqLess * 100) / totalAtom

        self.mean = Float(mean)
        self.meanA = Float(meanA)
        self.mean2 = Float(mean2)
        self.meanA2 = Float(meanA2)
        self.total_atom = Integer(totalAtom)
        self.fscq_greater = Integer(fscqGreater)
        self.fscq_less = Integer(fscqLess)
        self.porc_greater = Float(porcGreater)
        self.porc_less = Float(porcLess)
        self._store()

    def getFscqAttrDic(self):
        fscqDic = {}

        # Reading FCSQ's value file
        atStHandler = AtomicStructHandler()
        atStHandler.read(self.getFSCQFile())

        # Reading value stored in occupancy field for each atom
        for model in atStHandler.structure:
            for atom in model.get_atoms():
                fId = atom.get_full_id()
                chainName, resNumber, atomName = fId[2], fId[3][1], fId[4][0]
                resId = '{}:{}@{}'.format(chainName, resNumber, atomName)
                fscqDic[resId] = atom.get_occupancy() # FCSQ value (stored in occupancy variable)

        return fscqDic

    def createOutputStep(self):
        self._setMetrics()
        fscqDic = self.getFscqAttrDic()

        AS = self.getInputStruct()
        ASCIF = self.getInputStructCIF()
        ASH = AtomicStructHandler()
        cifDic = ASH.readLowLevel(ASCIF)
        cifDic = addScipionAttribute(cifDic, fscqDic, self._ATTRNAME, recipient='atoms')
        ASH._writeLowLevel(self._getPath(OUTPUT_CIF), cifDic)

        outAS = AS.clone()
        outAS.setFileName(self._getPath(OUTPUT_CIF))
        outAS.setVolume(self.inputVolume.get())

        self._defineOutputs(outputAtomStruct=outAS)
        if not self.fromFile:
            self._defineTransformRelation(self.inputPDBObj, outAS)


    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + VALIDATE_METHOD_URL)
        return messages

    def _summary(self):
        summary = []
        summary.append("Deviation from the signal of the Half Maps")
        if self.hasAttribute('mean'):
            summary.append("Mean FSC-Q: %.2f" % (self.mean.get()))
        if self.hasAttribute('meanA'):
            summary.append("Absotute Mean FSC-Q: %.2f" % (self.meanA.get()))

        summary.append(" ")
        summary.append("Deviation from the signal of the Half Maps divided by local resolution")
        if self.hasAttribute('mean2'):
            summary.append("Mean FSC-Qr: %.2f" % (self.mean2.get()))
        if self.hasAttribute('meanA2'):
            summary.append("Absotute Mean FSC-Qr: %.2f" % (self.meanA2.get()))

        summary.append("------------------------------------------")
        if self.hasAttribute('total_atom'):
            summary.append("Total number of atoms analyzed: %d" % (self.total_atom.get()))
        if (self.hasAttribute('fscq_greater') and self.hasAttribute('porc_greater')):
            summary.append("Number of atoms with FSC-Q>0.5: %d.  Percentage of total: %.2f."
                       % (self.fscq_greater.get(), self.porc_greater.get()))
        if (self.hasAttribute('fscq_less') and self.hasAttribute('porc_less')):
            summary.append("Number of atoms with FSC-Q<-0.5: %d.  Percentage of total: %.2f."
                       % (self.fscq_less.get(), self.porc_less.get()))
        return summary

    def _validate(self):
        errors = []
        if self.inputVolume.hasValue():
            #FIX CSVList:
            # volume.hasHalfMaps() does not work unless you call
            # getHalfMaps() or something else that triggers the CSVList.get()
            # that, populates the objValue. Just a print vol.getHalfMaps() will
            # change the behaviour of hasValue()
            # To review when migrating to Scipion3
            if not self.inputVolume.get().getHalfMaps():
                errors.append("Input Volume needs to have half maps. "
                "If you have imported the volume, be sure to import the half maps.")

        if self.fromFile and not self.inputPDB.get():
            errors.append('You have to provide a PDB file as input')
        elif not self.fromFile and not self.inputPDBObj.get():
            errors.append('You have to provide a PDB object as input')

        try:
            import bsoft
            if bsoft.__version__ in ["3.0.0", "3.0.1", "3.0.4"]:
                errors.append("This protocol requires bsoft plugin 3.0.5 or above to run."
                              " You have %s. Update it using the plugin manager or command line" % bsoft.__version__)
        except Exception as e:
            errors.append("This protocol requires bsoft plugin 3.0.5 or above to run. Update it using the plugin manager or command line")

        return errors

    def _citations(self):
        return ['Ramirez-Aportela 2020']

    def getInputStructFile(self):
        if self.fromFile:
            return self.inputPDB.get()
        else:
            return self.inputPDBObj.get().getFileName()
    
    def getInputStructCIF(self) -> str:
        """
        ### This function returns the full path for the input Atom Struct file in cif format.
        #### If file is not in cif format, it is converted to it. 

        #### Returns:
        - (str): Input Atom Struct's file's path.
        """
        # Get raw input filename
        structFile = self.getInputStructFile()

        if '.cif' not in structFile:
            # If file is not in cif format, convert it
            structFile = toCIF(structFile, self._getTmpPath('inputStruct.cif'))
        elif structFile.endswith('.cif.gz'):
            # If file is cif but compressed as a gz, extract it
            oFile = self._getTmpPath('inputStruct.cif')
            with gzip.open(structFile, 'rb') as fIn:
                with open(oFile, 'wb') as fOut:
                    shutil.copyfileobj(fIn, fOut)
            structFile = oFile
        
        # Return input cif file
        return structFile

    def getStructFile(self) -> str:
        """
        ### This function returns the full path for the Atom Struct file.

        #### Returns:
        - (str): Atom Struct's file's path.
        """
        return self._getExtraPath('inputStruct' + self.getStructExtension())

    def getInputStruct(self):
        if self.fromFile:
            return AtomStruct(filename=self.inputPDB.get())
        else:
            return self.inputPDBObj.get()

    def isStructExtensionValid(self) -> bool:
        """
        ### This function returns True if the input struct extension is one of the accepted types.

        #### Returns:
        - (bool): True if the input struct has an accepted extension. False otherwise.
        """
        # Getting file's natural extension
        extension = os.path.splitext(self.getInputStructFile())[-1]

        # Return extension validity
        return extension == '.pdb' or extension == '.ent' or extension == '.cif' or extension == ".mmcif" or extension == ".cif.gz"
    
    def getStructExtension(self) -> str:
        """
        ### This function returns the expected extension for the Atom Struct file.

        #### Returns:
        - (str): Atom Struct's file's extension.
        """
        # Getting file's natural extension
        extension = os.path.splitext(self.getInputStructFile())[-1]

        # If extension is a maintainable type, return as is (not all valid types are maintainable)
        # .pdb is not considered "maintainable" type as it is the default
        if extension == '.cif' or extension == ".mmcif" or extension == ".cif.gz":
            return extension
        
        # Return .pdb by default
        return '.pdb'
    
    def getFSCQFile(self) -> str:
        """
        ### This function returns the filename for FSCQ's value file.

        #### Returns:
        - (str): FSCQ's value file's full name with extension.
        """
        return self._getExtraPath(f'pdb_fsc-q{self.getStructExtension()}')

    def getNormFSCQFile(self) -> str:
        """
        ### This function returns the filename for FSCQ's norm file.

        #### Returns:
        - (str): FSCQ's norm file's full name with extension.
        """
        return self._getExtraPath(f'pdb_fsc-q_norm{self.getStructExtension()}')
