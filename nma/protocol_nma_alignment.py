# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Qiyu Jin
# *           Slavica Jonic                (jonic@impmc.upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Jan 2014
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************

import math
from glob import glob

from pyworkflow.em import *  
from pyworkflow.utils import * 
from pyworkflow.protocol.constants import LEVEL_EXPERT, LEVEL_ADVANCED
from pyworkflow.em.packages.xmipp3.convert import writeSetOfParticles
from pyworkflow.protocol.params import NumericRangeParam
import xmipp 

NMA_ALIGNMENT_WAV = 0
NMA_ALIGNMENT_PROJ = 1    

        
class XmippProtAlignmentNMA(ProtAnalysis3D):
    """ Protocol for flexible angular alignment. """
    _label = 'nma analysis'
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('modeList', NumericRangeParam, label="Modes selection",
                      help='Select which modes do you want to use from all of them.\n'
                           'If you leave the field empty, all modes will be used.\n'
                           'You have several ways to specify selected modes.\n'
                           '   Examples:\n'
                           ' "1,5-8,10" -> [1,5,6,7,8,10]\n'
                           ' "2,6,9-11" -> [2,6,9,10,11]\n'
                           ' "2 5, 6-8" -> [2,5,6,7,8])\n')
        form.addParam('inputParticles', PointerParam, label="Input particles", 
                      pointerClass='SetOfParticles',
                      help='Select the set of particles that you want to use for flexible analysis.')  
        form.addParam('inputPdb', PointerParam, label="Input PDB",  
                      pointerClass='PdbFile',
                      help='Atomic or pseudo-atomic structure to apply the normal modes.')
        form.addParam('inputModes', PointerParam, label="Normal modes",  
                      pointerClass='NormalModes',
                      help='Set of normal modes to explore.')
        
        form.addParam('copyDeformations', StringParam,
                      expertLevel=LEVEL_EXPERT,
                      label='Copy deformations(only for debug)')
        
        form.addSection(label='Angular assignment and mode detection')
        form.addParam('trustRegionScale', IntParam, default=1,
                      expertLevel=LEVEL_ADVANCED,
                      label='Trust region scale',
                      help='For elastic alignment, this parameter scales the initial \n'
                           'value of the trust region radius for optimization purposes.\n'
                           'Use larger values for larger expected deformation amplitudes.')    
        form.addParam('alignmentMethod', EnumParam, default=NMA_ALIGNMENT_WAV,
                      choices=['wavelets & splines', 'projection matching'],
                      label='Alignment method',
                      help='For rigid-body alignment, use Projection Matching (faster) instead\n'
                           'of Wavelets and Splines (more accurate). In the case of Wavelets \n'
                           'and Splines, the size of images should be a power of 2.')
        form.addParam('discreteAngularSampling', FloatParam, default=10,
                      label="Discrete angular sampling (deg)", 
                      help='This parameter is used in Projection Matching and Wavelets methods\n'
                           'for a rough rigid-body alignment. It is the angular step (in degrees)\n'
                           'with which the library of reference projections is computed. This \n'
                           'alignment is refined with Splines method if Wavelets and Splines \n'
                           'alignment is chosen.')
                      
        form.addParallelSection(threads=1, mpi=8)    
    
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        atomsFn = self.inputPdb.get().getFileName()
        modesFn = self.inputModes.get().getFileName()
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('images.xmd') 
        self._insertFunctionStep('convertInputStep') 
        
        localModesFn = self._getBasePath(modesFn)
        
        self._insertFunctionStep('copyFilesStep', atomsFn, modesFn)
        self._insertFunctionStep('selectModesStep', localModesFn)
        
        if self.copyDeformations.empty(): #ONLY FOR DEBUGGING
            self._insertFunctionStep("performNmaStep", self._getBasePath(atomsFn), localModesFn)
            self._insertFunctionStep("extractDeformationsStep")
        else:            
            self._insertFunctionStep('copyDeformationsStep', self.copyDeformations.get())
            
        self._insertFunctionStep('createOutputStep')
        
        
    #--------------------------- STEPS functions --------------------------------------------   
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(),self.imgsFn)
         
    def copyFilesStep(self, atomsFn, modesFn):
        """ Copy the input files to the local working dir. """
        for fn in [modesFn, atomsFn]:
            copyFile(fn, self._getBasePath(fn))
            
    def selectModesStep(self, modesFn):
        """ Read the modes metadata and keep only those modes selected
        by the user in the protocol. 
        """
        if not self.modeList.empty():
            modeList = getListFromRangeString(self.modeList.get())
            md = xmipp.MetaData(modesFn)
            
            for objId in md:
                order = md.getValue(xmipp.MDL_ORDER, objId)
                if order in modeList:
                    enable = 1
                else:
                    enable = 0
                md.setValue(xmipp.MDL_ENABLED, enable, objId)
            
            md.write(modesFn)
            
    def copyDeformationsStep(self, defFn):
        copyFile(defFn, self._getExtraPath(basename(defFn)))
        
    def performNmaStep(self, atomsFn, modesFn):
        sampling = self.inputParticles.get().getSamplingRate()
        discreteAngularSampling = self.discreteAngularSampling.get()
        trustRegionScale = self.trustRegionScale.get()
        odir = self._getTmpPath()
        imgFn = self.imgsFn
        
        args = "-i %(imgFn)s --pdb %(atomsFn)s --modes %(modesFn)s --sampling_rate %(sampling)f "
        args += "--discrAngStep %(discreteAngularSampling)f --odir %(odir)s --centerPDB "
        args += "--trustradius_scale %(trustRegionScale)d --resume "
        
        if self.inputPdb.get().getPseudoAtoms():
            args += "--fixed_Gaussian "
        
        if self.alignmentMethod == NMA_ALIGNMENT_PROJ:
            args += "--projMatch "
    
        print "self.numberOfMpi", self.numberOfMpi
        self.runJob("xmipp_nma_alignment", args % locals(), self.numberOfMpi.get())
        cleanPath(self._getPath('nmaTodo.xmd'))
    
    def extractDeformationsStep(self):
        md = xmipp.MetaData(self.imgsFn)
        deformations = md.getColumnValues(xmipp.MDL_NMA)
        defFn = self._getExtraPath("deformations.txt")
        fhDef = open(defFn, 'w')
        for deformation in deformations:
            for coef in deformation:
                fhDef.write("%f " % coef)
            fhDef.write("\n")
        fhDef.close()
        return defFn
    
    def createOutputStep(self):
        modes = NormalModes(filename=self._getLocalModesFn())
        self._defineOutputs(selectedModes=modes)

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary
    
    def _validate(self):
        errors = []
        xdim = self.inputParticles.get().getDimensions()[0]
        if not isPower2(xdim):
            errors.append("Image dimension (%s) is not a power of two, consider resize them" % xdim)
        return errors
    
    def _citations(self):
        return ['Jonic2005', 'Sorzano2004b']
    
    def _methods(self):
        pass
    
    #--------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd', 
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd",'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    def _getLocalModesFn(self):
        modesFn = self.inputModes.get().getFileName()
        return self._getBasePath(modesFn)
    
