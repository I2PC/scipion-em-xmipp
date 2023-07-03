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
from pyworkflow.utils import replaceBaseExt, removeExt, getExt, createLink, replaceExt

from pwem.convert import cifToPdb, downloadPdb, headers
from pwem.objects import Volume, Transform, SetOfVolumes, SetOfAtomStructs, AtomStruct
from pwem.protocols import ProtInitialVolume


class XmippProtConvertPdb(ProtInitialVolume):
    """ Convert atomic structure(s) into volume(s) """
    _label = 'convert pdb to volume'
    OUTPUT_NAME1 = "outputVolume"
    OUTPUT_NAME2 = "outputVolumes"
    _possibleOutputs = {OUTPUT_NAME1: Volume, OUTPUT_NAME2: SetOfVolumes}
       
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        """ Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        form.addSection(label='Input')
        form.addParam('pdbObj', params.PointerParam, pointerClass='AtomStruct, SetOfAtomStructs',
                      label="Input structure(s) ",
                      help='Specify input atomic structure(s).')
        form.addParam('sampling', params.FloatParam, default=1.0,
                      label="Sampling rate (Å/px)",
                      help='Sampling rate (Angstroms/pixel)')
        form.addParam('vol', params.BooleanParam, label='Use a volume as an empty template?', default=False,
                      condition='not isinstance(pdbObj, SetOfAtomStructs)',
                      help='Use an existing volume to define the size and origin for the output volume. If this option'
                           'is selected, make sure that "Center PDB" in advanced parameters is set to *No*.')
        form.addParam('volObj', params.PointerParam, pointerClass='Volume',
                      label="Input volume ", condition='not isinstance(pdbObj, SetOfAtomStructs) and vol', allowsNull=True,
                      help='The origin and the final size of the output volume will be taken from this volume.')
        form.addParam('setSize', params.BooleanParam, label='Set final size?', default=False,
                      condition='not vol and not isinstance(pdbObj, SetOfAtomStructs)')
        form.addParam('size', params.IntParam,
                      label="Box side size (px)", condition='isinstance(pdbObj, SetOfAtomStructs)', allowsNull=True,
                      help='This size should apply to all volumes')
        form.addParam('size_z', params.IntParam, condition='setSize and not vol', allowsNull=True,
                      label="Final size (px) Z",
                      help='Final size in Z in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_y', params.IntParam, condition='setSize and not vol', allowsNull=True,
                      label="Final size (px) Y",
                      help='Final size in Y in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_x', params.IntParam, condition='setSize and not vol', allowsNull=True,
                      label="Final size (px) X",
                      help='Final size in X in pixels. If desired output size is x = y = z you can only fill this '
                           'field. If no value is provided, protocol will estimate it.')
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
        if isinstance(self.pdbObj.get(), AtomStruct):
            self._insertFunctionStep(self.convertPdbStep)
        else:
            self._insertFunctionStep(self.convertPdbSetStep)
        self._insertFunctionStep(self.createOutput)

    # --------------------------- STEPS functions --------------------------------------------
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

        if " " in pdbFn:
            pdbFn_extra = self._getExtraPath(os.path.basename(pdbFn.replace(" ", "_")))
        else:
            pdbFn_extra = self._getExtraPath(os.path.basename(pdbFn))

        createLink(pdbFn, pdbFn_extra)

        samplingR = self.sampling.get()

        args = '-i "%s" --sampling %f -o "%s"' % (pdbFn_extra, samplingR, outFile)
        
        if self.centerPdb:
            args += ' --centerPDB'
            if self.outPdb:
                args += ' --oPDB'

        if self.vol:
            vol = self.volObj.get()
            size = vol.getDim()
            ccp4header = headers.Ccp4Header(vol.getFileName(), readHeader=True)
            self.shifts = ccp4header.getOrigin()
            args += ' --size %d %d %d --orig %d %d %d' % (size[2], size[1], size[0],
                                                          self.shifts[0]/samplingR,
                                                          self.shifts[1]/samplingR,
                                                          self.shifts[2]/samplingR)

        if self.setSize and not self.vol:
            args += ' --size'

            if self.size_x.hasValue():
                args += ' %d' % self.size_x.get()

            if self.size_y.hasValue() and self.size_z.hasValue():
                args += ' %d %d' % (self.size_y.get(), self.size_z.get())

        self.info("Input file: " + pdbFn)
        self.info("Output file: " + outFile)
        
        program = "xmipp_volume_from_pdb"
        self.runJob(program, args)


    def convertPdbSetStep(self):
        """ A function to loop over a set of atomic structures and converts each to volume
        """
        pdbFns = self._getPdbFileName()
        samplingR = self.sampling.get()

        for pdbFn in pdbFns:
            if getExt(pdbFn) == ".cif":
                pdbFn2 = self._getExtraPath(replaceBaseExt(pdbFn, 'pdb'))
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
        if isinstance(self.pdbObj.get(), AtomStruct):
            volume = Volume()
            volume.setSamplingRate(self.sampling.get())
            # Since xmipp generates always a .vol we do the conversion here
            ih = ImageHandler()
            mrcFn = self._getVolName(extension="mrc")
            ih.convert(self._getVolName(), mrcFn)
            volume.setFileName(mrcFn)
            volume.fixMRCVolume(setSamplingRate=True)
            setMRCSamplingRate(mrcFn, self.sampling.get())
            if self.vol:
                origin = Transform()
                origin.setShiftsTuple(self.shifts)
                volume.setOrigin(origin)
            self._defineOutputs(**{self.OUTPUT_NAME1:volume})
            self._defineSourceRelation(self.pdbObj, volume)
        else:  # case of a set of atomic structures as input
            volumes = self._createSetOfVolumes()
            volumes.setSamplingRate(self.sampling.get())
            pdbFns = self._getPdbFileName()
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
            self._defineOutputs(**{self.OUTPUT_NAME2: volumes})
            self._defineSourceRelation(self.pdbObj, volumes)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        """ Even if the full set of parameters is available, this function provides
        summary information about an specific run.
        """ 
        summary = []
        # Add some lines of summary information
        if not (hasattr(self, self.OUTPUT_NAME1) or hasattr(self, self.OUTPUT_NAME2)):
            summary.append("The output is not ready yet.")
        return summary
      
    def _validate(self):
        """ The function of this hook is to add some validation before the protocol
        is launched to be executed. It should return a list of errors. If the list is
        empty the protocol can be executed.
        """
        errors = []
        return errors
    
    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileName(self):
        if isinstance(self.pdbObj.get(), AtomStruct):
            return self.pdbObj.get().getFileName()
        else:
            return [i.getFileName() for i in self.pdbObj.get()]

    def _getVolName(self, extension="vol"):
        return self._getExtraPath(replaceBaseExt(self._getPdbFileName().replace(" ", "_"), extension))
