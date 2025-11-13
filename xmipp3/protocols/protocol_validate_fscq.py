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
import xmipp3
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
   Assesses the quality of the fit between a model and experimental data. This protocol evaluates how well a volume or structure matches reference data, guiding improvements in model accuracy.
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

        group.addParam('pdbMap', PointerParam, pointerClass='Volume',
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
        if self.pdbMap.hasValue():
            pdbvolume = self.pdbMap.get()
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

            if (not self.pdbMap.hasValue()):

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
        Vx = xmipp3.Image(self._getFileName(RESTA_FILE))
        V=Vx.getData()
        Vmask = xmipp3.Image(self._getFileName(MASK_FILE_MRC)+':mrc').getData()
        Vres = xmipp3.Image(self._getFileName(BLOCRES_HALF_FILE)+':mrc').getData()
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
