# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Jesus Cuenca (jcuenca@cnb.csic.es)
# *           Roberto Marabini (rmarabini@cnb.csic.es)
# *           Ignacio Foche
# * 
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

import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as const
from pyworkflow.utils import replaceBaseExt, removeExt, getExt

from pwem.convert import cifToPdb, downloadPdb, headers
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume


class XmippProtConvertPdb(ProtInitialVolume):
    """ Convert a PDB file into a volume.  """
    _label = 'convert a PDB'
    IMPORT_FROM_ID = 0
    IMPORT_OBJ = 1
    IMPORT_FROM_FILES = 2 
       
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        """ Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        form.addSection(label='Input')
        form.addParam('inputPdbData', params.EnumParam, choices=['id', 'object', 'file'],
                      label="Retrieve PDB from", default=self.IMPORT_FROM_ID,
                      display=params.EnumParam.DISPLAY_HLIST,
                      help='Retrieve PDB data from server, use a pdb Object, or a local file')
        form.addParam('pdbId', params.StringParam, condition='inputPdbData == IMPORT_FROM_ID',
                      label="Pdb Id ", allowsNull=True,
                      help='Type a pdb Id (four alphanumeric characters).')
        form.addParam('pdbObj', params.PointerParam, pointerClass='AtomStruct',
                      label="Input pdb ", condition='inputPdbData == IMPORT_OBJ', allowsNull=True,
                      help='Specify a pdb object.')
        form.addParam('pdbFile', params.FileParam,
                      label="File path", condition='inputPdbData == IMPORT_FROM_FILES', allowsNull=True,
                      help='Specify a path to desired PDB structure.')
        form.addParam('sampling', params.FloatParam, default=1.0, 
                      label="Sampling rate (Å/px)",
                      help='Sampling rate (Angstroms/pixel)')
        form.addParam('vol', params.BooleanParam, label='Use a volume as an empty template?', default=False,
                      help='Use an existing volume to define the size and origin for the output volume. If this option'
                           'is selected, make sure that "Center PDB" in advanced parameters is set to *No*.')
        form.addParam('volObj', params.PointerParam, pointerClass='Volume',
                      label="Input volume ", condition='vol', allowsNull=True,
                      help='The origin and the final size of the output volume will be taken from this volume.')
        form.addParam('setSize', params.BooleanParam, label='Set final size?', default=False, condition='vol == False')
        form.addParam('size_z', params.IntParam, condition='setSize', allowsNull=True,
                      label="Final size (px) Z",
                      help='Final size in Z in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_y', params.IntParam, condition='setSize', allowsNull=True,
                      label="Final size (px) Y",
                      help='Final size in Y in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_x', params.IntParam, condition='setSize', allowsNull=True,
                      label="Final size (px) X",
                      help='Final size in X in pixels. If desired output size is x = y = z you can only fill this '
                           'field. If no value is provided, protocol will estimate it.')
        form.addParam('centerPdb', params.BooleanParam, default=True, 
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Center PDB",
                      help='Center PDB with the center of mass')
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ In this function the steps that are going to be executed should
        be defined. Two of the most used functions are: _insertFunctionStep or _insertRunJobStep
        """
        if self.inputPdbData == self.IMPORT_FROM_ID:
            self._insertFunctionStep('pdbDownloadStep')
        self._insertFunctionStep('convertPdbStep')
        self._insertFunctionStep('createOutput')
    
    # --------------------------- STEPS functions --------------------------------------------
    def pdbDownloadStep(self):
        """Download all pdb files in file_list and unzip them."""
        downloadPdb(self.pdbId.get(), self._getPdbFileName(), self._log)
        
    def convertPdbStep(self):
        """ Although is not mandatory, usually is used by the protocol to
        register the resulting outputs in the database.
        """
        pdbFn = self._getPdbFileName()
        outFile = removeExt(self._getVolName())
        if getExt(pdbFn)==".cif":
            pdbFn2=replaceBaseExt(pdbFn, 'pdb')
            cifToPdb(pdbFn, pdbFn2)
            pdbFn = pdbFn2

        samplingR = self.sampling.get()

        args = '-i %s --sampling %f -o %s' % (pdbFn, samplingR, outFile)
        
        if self.centerPdb:
            args += ' --centerPDB'

        if self.vol:
            vol = self.volObj.get()
            size = vol.getDim()
            ccp4header = headers.Ccp4Header(vol.getFileName(), readHeader=True)
            self.shifts = ccp4header.getOrigin()
            args += ' --size %d %d %d --orig %d %d %d' % (size[2], size[1], size[0],
                                                          self.shifts[0]/samplingR,
                                                          self.shifts[1]/samplingR,
                                                          self.shifts[2]/samplingR)

        if self.setSize:
            args += ' --size'

        if self.size_x.hasValue() and self.setSize:
            args += ' %d' % self.size_x.get()

        if self.size_y.hasValue() and self.size_z.hasValue() and self.setSize:
            args += ' %d %d' % (self.size_y.get(), self.size_z.get())

        self.info("Input file: " + pdbFn)
        self.info("Output file: " + outFile)
        
        program = "xmipp_volume_from_pdb"
        self.runJob(program, args)

    def createOutput(self):
        volume = Volume()
        volume.setSamplingRate(self.sampling.get())
        volume.setFileName(self._getVolName())
        if self.vol:
            origin = Transform()
            origin.setShiftsTuple(self.shifts)
            volume.setOrigin(origin)
        self._defineOutputs(outputVolume=volume)
        if self.inputPdbData == self.IMPORT_OBJ:
            self._defineSourceRelation(self.pdbObj, volume)
    
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        """ Even if the full set of parameters is available, this function provides
        summary information about an specific run.
        """ 
        summary = []
        # Add some lines of summary information
        if not hasattr(self, 'outputVolume'):
            summary.append("outputVolume not ready yet.")
        else:
            if self.inputPdbData == self.IMPORT_FROM_ID:
                summary.append("Input PDB ID: %s" % self.pdbId.get())
            elif self.inputPdbData == self.IMPORT_OBJ:
                summary.append("Input PDB File: %s" % self.pdbObj.get().getFileName())
            else:
                summary.append("Input PDB File: %s" % self.pdbFile.get())
        return summary
      
    def _validate(self):
        """ The function of this hook is to add some validation before the protocol
        is launched to be executed. It should return a list of errors. If the list is
        empty the protocol can be executed.
        """
        errors = []
        if self.inputPdbData == self.IMPORT_FROM_ID:
            lenStr = len(self.pdbId.get())
            if lenStr != 4:
                errors = ["Pdb id is composed only by four alphanumeric characters"]
        
        return errors
    
    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileName(self):
        if self.inputPdbData == self.IMPORT_FROM_ID:
            return self._getExtraPath('%s.cif' % self.pdbId.get())
        elif self.inputPdbData == self.IMPORT_OBJ:
            return self.pdbObj.get().getFileName()
        else:
            return self.pdbFile.get()
    
    def _getVolName(self):
        return self._getExtraPath(replaceBaseExt(self._getPdbFileName(), "vol"))
