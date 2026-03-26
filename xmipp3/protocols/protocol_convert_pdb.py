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

# General imports
import os

# Scipion em imports
from pwem.convert.headers import setMRCSamplingRate
from pwem.emlib.image import ImageHandler
from pwem.convert import cifToPdb, headers
from pwem.objects import Volume, Transform, SetOfVolumes, AtomStruct, SetOfAtomStructs
from pwem.protocols import ProtInitialVolume
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as const
from pyworkflow.utils import replaceBaseExt, removeExt, getExt, createLink, replaceExt, removeBaseExt, red
from pyworkflow import UPDATED, PROD

class XmippProtConvertPdb(ProtInitialVolume):
    """  Converts atomic structure files in PDB (Protein Data Bank) format into volumetric maps. Converting a PDB to a volume generates a simulated electron density map, useful for validating atomic models, fitting into experimental maps or performing docking."""
    _label = 'convert pdbs to volumes'
    _devStatus = PROD
    OUTPUT_NAME1 = "outputVolume"
    OUTPUT_NAME2 = "outputVolumes"
    OUTPUT_NAME3 = "outputPdb"
    OUTPUT_NAME4 = "outputAtomStructs"
    _possibleOutputs = {OUTPUT_NAME1: Volume, OUTPUT_NAME2: SetOfVolumes,
                        OUTPUT_NAME3: AtomStruct, OUTPUT_NAME4: SetOfAtomStructs}

    # --------------------------- Class constructor --------------------------------------------
    def __init__(self, **args):
        # Calling parent class constructor
        super().__init__(**args)

        # Defining execution mode. Steps will take place in parallel now
        # Full tutorial on how to parallelize protocols can be read here:
        # https://scipion-em.github.io/docs/release-3.0.0/docs/developer/parallelization.html
        self.stepsExecutionMode = params.STEPS_PARALLEL
       
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        """
        Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        # Defining condition string for x, y, z coords
        coordsCondition = 'setSize and not vol and not isinstance(pdbObj, SetOfAtomStructs)'

        # Defining parallel arguments
        form.addParallelSection(threads=4, mpi=1)
        form.addParam('binThreads', params.IntParam,
                      label='threads',
                      default=2,
                      help='Number of threads used by Xmipp each time it is called in the protocol execution. For '
                           'example, if 3 Scipion threads and 3 Xmipp threads are set, the pdbs will be '
                           'processed in groups of 2 at the same time with a call of Xmipp with 3 threads each, so '
                           '6 threads will be used at the same time. Beware the memory of your machine has '
                           'memory enough to load together the number of pdbs specified by Scipion threads.')

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
                      help='Center PDB with the center of mass.')
        form.addParam('outPdb', params.BooleanParam, default=False, 
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Store centered PDB",
                      help='Set to \'Yes\' if you want to save centered PDB. '
                           'It will be stored in the output directory of this protocol.')
        form.addParam('convertCif', params.BooleanParam, default=False,
                      expertLevel=const.LEVEL_ADVANCED, 
                      label="Convert CIF to PDB",
                      help='If set to true and input atom struct file is a CIF, it will get converted to PDB.')
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
        pdbList = self._getPdbFileNames()
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
        # Conditionally converting all input atomic structures to .pdb
        pdb = self._convertAtomStruct(pdb)
        
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
        extraPath = self._getExtraPath()

        # Saving centered pdbs if wanted
        if self.centerPdb and self.outPdb:
            centeredPdbs = self._getExtraPath('centerted')
            self.runJob('mkdir -p',  centeredPdbs)
            self.runJob('mv', '{}/*_centered.pdb {}'.format(extraPath, centeredPdbs))

        # Generating output objects
        outputVol = self._createSetOfVolumes() if isSet else None
        pdbOut = self.centerPdb and self.outPdb
        if pdbOut:
            outputPdb = self._createSetOfPDBs() if isSet else None

        # Instantiating image handler
        ih = ImageHandler()

        # Generating input list
        pdbFns = self._getPdbFileNames()

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
                outputVol.append(volume)
            # Generating output PDB
            if pdbOut:
                centeredPdbFn = '{}/{}_centered.pdb'.format(centeredPdbs, removeBaseExt(pdbFn))
                atom = AtomStruct(filename=centeredPdbFn)
                atom.setVolume(volume)
                if isSet:
                    outputPdb.append(atom)
        
        if not isSet:
            # If input is single atom struct, get produced volume as output
            outputVol = volume
            if self.vol:
                origin = Transform()
                origin.setShiftsTuple(self.shifts)
                outputVol.setOrigin(origin)

            if pdbOut:
                outputPdb = atom

        # Setting ouput sampling rate
        outputVol.setSamplingRate(samplingR)

        # Removing temporary files
        if self.clean:
            self.runJob('rm', '-rf {}/*.vol {}/*.cif {}/*.pdb'.format(extraPath, extraPath, extraPath))

        if pdbOut:
            outputPdbName = self.OUTPUT_NAME4 if isSet else self.OUTPUT_NAME3
            self._defineOutputs(**{outputPdbName: outputPdb})

        # Defining output
        outputVolName = self.OUTPUT_NAME2 if isSet else self.OUTPUT_NAME1
        self._defineOutputs(**{outputVolName: outputVol})
        self._defineSourceRelation(self.pdbObj, outputVol)

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

        # Checking if MPI is selected (only threads are allowed)
        if self.numberOfMpi > 1:
            errors.append('MPI cannot be selected, because Scipion is going to drop support for it. Select threads instead.')

        if isinstance(self.pdbObj.get(), SetOfAtomStructs) and not self.size.hasValue():
            errors.append('Please set size when using SetOfAtomStructs as input')

        return errors
    
    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileNames(self):
        """ This function returns the input file/s stored in a list. """
        pbdObj = self.pdbObj.get()
        if isinstance(pbdObj, AtomStruct):
            # If it is a single AtomStruct, place it inside a list of 1 element
            return [pbdObj.getFileName()]
        else:
            # If it is a SetOfAtom Structs, get all of the elements iterating the set
            return [i.getFileName() for i in pbdObj]

    def _convertAtomStruct(self, pdbRaw):
        """ This function receives an atomic struct file, and conditionally converts it to .pdb if it is in .cif format. """
        # Get output path for atomic struct file
        baseExt = replaceBaseExt(pdbRaw, 'pdb') if self.convertCif.get() else os.path.basename(pdbRaw)
        convertedPdb = self._getExtraPath(baseExt).replace(" ", "_")

        # If file extension is .cif and conversion is required, convert to .pdb, or else we link it as is
        if getExt(pdbRaw) == ".cif" and self.convertCif.get():
            try:
                cifToPdb(pdbRaw, convertedPdb)
            except:
                self.error(red("The conversion has failed. It might be for an excessive number of atoms. "\
                          "Try not to convert the CIF into a PDB (hidden flag)."))
                raise RuntimeError("PDB conversion")
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