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

# Scipion em imports
from pwem.convert.headers import setMRCSamplingRate
from pwem.emlib.image import ImageHandler
from pwem.convert import cifToPdb, headers
from pwem.objects import Volume, Transform, SetOfVolumes, AtomStruct
from pwem.protocols import ProtInitialVolume
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as const
from pyworkflow.utils import replaceBaseExt, removeExt, getExt, createLink, replaceExt

class XmippProtConvertPdb(ProtInitialVolume):
    """ Convert atomic structure(s) into volume(s) """
    _label = 'convert pdbs to volumes'
    OUTPUT_NAME1 = "outputVolume"
    OUTPUT_NAME2 = "outputVolumes"
    _possibleOutputs = {OUTPUT_NAME1: Volume, OUTPUT_NAME2: SetOfVolumes}

    # --------------------------- Class constructor --------------------------------------------
    def __init__(self, **args):
        # Calling parent class constructor
        super().__init__(**args)

        # Defining execution mode. Steps will take place in parallel now
        self.stepsExecutionMode = params.STEPS_PARALLEL
       
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        """
        Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        # Defining condition string for x, y, z coords
        coordsCondition = 'setSize and not vol'

        # Defining parallel arguments
        form.addParallelSection(threads=4)

        # Generating form
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
        form.addParam('size_z', params.IntParam, condition=coordsCondition, allowsNull=True, label="Final size (px) Z",
                      help='Final size in Z in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_y', params.IntParam, condition=coordsCondition, allowsNull=True, label="Final size (px) Y",
                      help='Final size in Y in pixels. If no value is provided, protocol will estimate it.')
        form.addParam('size_x', params.IntParam, condition=coordsCondition, allowsNull=True, label="Final size (px) X",
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
        form.addParam('clean', params.BooleanParam, default=True,
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Clean tmp files",
                      help='Delete all non-output files once the protocol has finished producing them.')
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """
        In this function the steps that are going to be executed should
        be defined. Two of the most used functions are: _insertFunctionStep or _insertRunJobStep
        """
        # Checking if input is set or not
        isSet = not isinstance(self.pdbObj.get(), AtomStruct)

        # Generating pdb list and obtaining input sampling rate
        pdbList = self._getPdbFileName() if isSet else [self._getPdbFileName()]
        samplingR = self.sampling.get()

        # Defining list of function ids to be waited by the createOutput function
        deps = []
        for pdbFn in pdbList:
            # Calling processConversion in parallel with each input data
            deps.append(self._insertFunctionStep(self.processConversion, pdbFn, samplingR, isSet, prerequisites=[]))
        
        # Insert output conversion step
        self._insertFunctionStep(self.createOutput, isSet, samplingR, prerequisites=deps)

    # --------------------------- STEPS functions --------------------------------------------
    def processConversion(self, pdb, samplingR, isSet):
        """ This step runs the pdb conversion. """
        # Converting all input atomic structures to .pdb
        pdb = self._convertToPdb(pdb)
        
        # Generating output file for each input
        outFile = removeExt(pdb)

        # Getting args for program call
        args = self._getConversionArgs(pdb, samplingR, outFile, isSet=isSet)

        # Showing input and output file names
        self.info("Input file: " + pdb)
        self.info("Output file: " + outFile)

        # Calling program
        self.runJob("xmipp_volume_from_pdb", args)

    def createOutput(self, isSet, samplingR):
        # Generating output object
        outputObj = self._createSetOfVolumes() if isSet else None

        # Instantiating image handler
        ih = ImageHandler()

        # Generating input list (only one element for non-set inputs)
        pdbFns = self._getPdbFileName() if isSet else [self._getVolName()]

        # Converting volumes. Since xmipp generates always a .vol we do the conversion here
        for pdbFn in pdbFns:
            # Generating ouput mrc
            volumeFn = self._getExtraPath(replaceBaseExt(pdbFn, 'vol'))
            mrcFn = replaceExt(volumeFn, 'mrc')
            ih.convert(volumeFn, mrcFn)
            setMRCSamplingRate(mrcFn, samplingR)
            # Generating output volume object from mrc
            volume = Volume()
            volume.setSamplingRate(samplingR)
            volume.setFileName(mrcFn)
            volume.fixMRCVolume(setSamplingRate=True)
            # If input is set, append volume to set
            if isSet:
                outputObj.append(volume)
        
        if not isSet:
            # If input is single atom struct, get produced volume as output
            outputObj = volume
            if self.vol:
                origin = Transform()
                origin.setShiftsTuple(self.shifts)
                outputObj.setOrigin(origin)

        # Setting ouput sampling rate
        outputObj.setSamplingRate(samplingR)

        # Removing temporary files
        if self.clean:
            outputPath = self._getExtraPath()
            self.runJob('rm', '-rf {}/*.vol {}/*.cif {}/*.pdb'.format(outputPath, outputPath, outputPath))

        # Defining output
        outputName = self.OUTPUT_NAME2 if isSet else self.OUTPUT_NAME1
        self._defineOutputs(**{outputName: outputObj})
        self._defineSourceRelation(self.pdbObj, outputObj)

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
    
    def _convertToPdb(self, pdbRaw):
        """ This function receives an atomic struct file, and converts it to .pdb if it is in .cif format. """
        # Get output path for pdb file
        convertedPdb = self._getExtraPath(replaceBaseExt(pdbRaw, 'pdb')).replace(" ", "_")

        # If file extension is .cif, convert to .pdb, or else we link it as is
        if getExt(pdbRaw) == ".cif":
            cifToPdb(pdbRaw, convertedPdb)
        else:
            createLink(pdbRaw, convertedPdb)

        # Return optionally converted file
        return convertedPdb
    
    def _getConversionArgs(self, pdb, samplingRate, outputFile, isSet=False):
        """ This function generates the arguments to convert an input pdb to a volume. """
        # Main arguments
        args = '-i "%s" --sampling %f -o "%s"' % (pdb, samplingRate, outputFile)

        # Flag for centered pdbs
        if self.centerPdb:
            args += ' --centerPDB'
            # Flag for output pdbs
            if self.outPdb:
                args += ' --oPDB'
        
        # Setting size and origin if selected
        if isSet:
            args += ' --size %d' % (self.size.get())
        else:
            if self.vol:
                vol = self.volObj.get()
                size = vol.getDim()
                ccp4header = headers.Ccp4Header(vol.getFileName(), readHeader=True)
                self.shifts = ccp4header.getOrigin()
                args += ' --size %d %d %d --orig %d %d %d' % (size[2], size[1], size[0],
                                                            self.shifts[0]/samplingRate,
                                                            self.shifts[1]/samplingRate,
                                                            self.shifts[2]/samplingRate)
            else:
                if self.setSize:
                    args += ' --size'

                    if self.size_x.hasValue():
                        args += ' %d' % self.size_x.get()

                    if self.size_y.hasValue() and self.size_z.hasValue():
                        args += ' %d %d' % (self.size_y.get(), self.size_z.get())
        
        # Returning produced args
        return args