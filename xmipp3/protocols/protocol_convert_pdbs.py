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

import os

import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as const
from pwem.convert.headers import setMRCSamplingRate
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import replaceBaseExt, removeExt, getExt, replaceExt

from pwem.convert import cifToPdb
from pwem.objects import SetOfVolumes, Volume
from pwem.protocols import ProtInitialVolume


class XmippProtConvertPdbs(ProtInitialVolume):
    """ Convert a set of PDBs to a set of volumes  """
    _label = 'convert a set of PDBs'
    OUTPUT_NAME = "outputVolumes"
    _possibleOutputs = {OUTPUT_NAME: SetOfVolumes}
       
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        """ Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        form.addSection(label='Input')
        form.addParam('pdbObj', params.PointerParam, pointerClass='SetOfAtomStructs',
                      label="Input pdbs ",
                      help='Specify a SetOfAtomicStructs object.')
        form.addParam('sampling', params.FloatParam, default=1.0, 
                      label="Sampling rate (Å/px)",
                      help='Sampling rate (Angstroms/pixel)')
        form.addParam('size', params.IntParam,
                      label="Box side size (px)",
                      help='This size should apply to all volumes')
        form.addParam('centerPdb', params.BooleanParam, default=True,
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Center PDB",
                      help='Center PDB with the center of mass')
        form.addParam('outPdb', params.BooleanParam, default=False, 
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Store centered PDB",
                      help='Set to \'Yes\' if you want to save centered PDB. '
                           'It will be stored in the output directory of this protocol')
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ In this function the steps that are going to be executed should
        be defined. Two of the most used functions are: _insertFunctionStep or _insertRunJobStep
        """
        self._insertFunctionStep(self.convertPdbStep)
        self._insertFunctionStep(self.createOutput)
    
    # --------------------------- STEPS functions --------------------------------------------
        
    def convertPdbStep(self):
        """ Although is not mandatory, usually is used by the protocol to
        register the resulting outputs in the database.
        """
        pdbFns = self._getPdbFileNames()
        samplingR = self.sampling.get()

        for pdbFn in pdbFns:
            if getExt(pdbFn)==".cif":
                pdbFn2= self._getExtraPath(replaceBaseExt(pdbFn, 'pdb'))
                cifToPdb(pdbFn, pdbFn2)
                pdbFn = pdbFn2

            outFile = self._getExtraPath(os.path.basename(removeExt(pdbFn)))
            args = '-i {} --sampling {} -o {}'.format(pdbFn, samplingR, outFile)
        
            if self.centerPdb:
                args += ' --centerPDB'
                if self.outPdb:
                    args += ' --oPDB'

                args += ' --size {}'.format(self.size.get())

            program = "xmipp_volume_from_pdb"
            self.runJob(program, args)

    def createOutput(self):
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self.sampling.get())

        # Since xmipp generates always a .vol we do the conversion here
        pdbFns = self._getPdbFileNames()
        for pdbFn in pdbFns:
            volumefn = self._getExtraPath(replaceBaseExt(pdbFn, 'vol'))
            ih = ImageHandler()
            mrcFn = replaceExt(volumefn, 'mrc')
            ih.convert(volumefn, mrcFn)
            volume = Volume()
            volume.setSamplingRate(self.sampling.get())
            volume.setFileName(mrcFn)
            volume.fixMRCVolume(setSamplingRate=True)
            setMRCSamplingRate(mrcFn, self.sampling.get())
            volumes.append(volume)
        self._defineOutputs(**{self.OUTPUT_NAME:volumes})
        self._defineSourceRelation(self.pdbObj, volumes)
    
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        """ Even if the full set of parameters is available, this function provides
        summary information about an specific run.
        """ 
        summary = []
        # Add some lines of summary information
        return summary
      
    def _validate(self):
        """ The function of this hook is to add some validation before the protocol
        is launched to be executed. It should return a list of errors. If the list is
        empty the protocol can be executed.
        """
        errors = []
        return errors
    
    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileNames(self):
        return [i.getFileName() for i in self.pdbObj.get()]

