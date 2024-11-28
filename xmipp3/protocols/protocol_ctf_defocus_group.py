# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
# *              J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
"""
This file implements CTF defocus groups using xmipp 3.1
"""

from math import pi

from pyworkflow.protocol.params import GE, PointerParam, FloatParam

from pwem.objects import SetOfDefocusGroup, DefocusGroup
from pwem.protocols import ProtProcessParticles


from pwem import emlib
from xmipp3.convert import writeSetOfParticles, writeSetOfDefocusGroups


# TODO: change the base class to a more apropiated one
class XmippProtCTFDefocusGroup(ProtProcessParticles):
    """
    Given a set of CTFs group them by defocus value.
    The output is a metadata file containing 
     a list of defocus values that delimite 
    each defocus group.
    """
    _label = 'defocus group'
    
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        """ Define the parameters that will be input for the Protocol.
        This definition is also used to generate automatically the GUI.
        """
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, 
                      pointerClass='SetOfParticles', pointerCondition='hasCTF',
                      label="Input particles with CTF", 
                      help='Select the input particles. \n '
                           'they should have information about the CTF (hasCTF=True)')
        form.addParam('ctfGroupMaxDiff', FloatParam, default=1,
                      label='Error for grouping', validators=[GE(1.,'Error must be greater than 1')],
                      help='Maximum error when grouping, the higher the more groups'
                           'This is a 1D program, only defocus U is used\n '
                           'The frequency at which the phase difference between the CTFs\n'
                           'belonging to 2 particles is equal to Pi/2 is computed \n '
                           'If this difference is less than 1/(2*factor*sampling_rate)\n' 
                           'then images are placed in different groups')          
        
    #--------------------------- INSERT steps functions --------------------------------------------  
    def _insertAllSteps(self):
        """ In this function the steps that are going to be executed should
        be defined. Two of the most used functions are: _insertFunctionStep or _insertRunJobStep
        """
        #TODO: when aggregation functions are defined in Scipion set
        # this step can be avoid and the protocol can remove Xmipp dependencies
        
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('images.xmd') 
        self._insertFunctionStep('convertInputStep') 
        
        ctfGroupMaxDiff = self.ctfGroupMaxDiff.get()
        
        #verifyFiles = []
        self._insertFunctionStep('createOutputStep', ctfGroupMaxDiff)
    
    #--------------------------- STEPS functions -------------------------------------------- 
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(),self.imgsFn)
              
    def createOutputStep(self, ctfGroupMaxDiff):
        """ Create defocus groups and generate the output set """
        fnScipion = self._getPath('defocus_groups.sqlite')
        fnXmipp   = self._getPath('defocus_groups.xmd')
        setOfDefocus = SetOfDefocusGroup(filename=fnScipion)
        df = DefocusGroup()
        mdImages    = emlib.MetaData(self.imgsFn)
        if not mdImages.containsLabel(emlib.MDL_CTF_SAMPLING_RATE):
            mdImages.setValueCol(emlib.MDL_CTF_SAMPLING_RATE,
                                 self.inputParticles.get().getSamplingRate())
        
        mdGroups    = emlib.MetaData()
        mdGroups.aggregateMdGroupBy(mdImages, emlib.AGGR_COUNT,
                                               [emlib.MDL_CTF_DEFOCUSU,
                                                emlib.MDL_CTF_DEFOCUS_ANGLE,
                                                emlib.MDL_CTF_SAMPLING_RATE,
                                                emlib.MDL_CTF_VOLTAGE,
                                                emlib.MDL_CTF_CS,
                                                emlib.MDL_CTF_Q0],
                                                emlib.MDL_CTF_DEFOCUSU,
                                                emlib.MDL_COUNT)
        mdGroups.sort(emlib.MDL_CTF_DEFOCUSU)
        
        mdCTFAux = emlib.MetaData()
        idGroup  = mdGroups.firstObject()
        idCTFAux = mdCTFAux.addObject()
        mdCTFAux.setValue(emlib.MDL_CTF_DEFOCUS_ANGLE, 0., idCTFAux);
        mdCTFAux.setValue(emlib.MDL_CTF_SAMPLING_RATE, mdGroups.getValue(emlib.MDL_CTF_SAMPLING_RATE,idGroup), idCTFAux)
        mdCTFAux.setValue(emlib.MDL_CTF_VOLTAGE,       mdGroups.getValue(emlib.MDL_CTF_VOLTAGE,idGroup),       idCTFAux)
        mdCTFAux.setValue(emlib.MDL_CTF_CS,            mdGroups.getValue(emlib.MDL_CTF_CS,idGroup),            idCTFAux)
        mdCTFAux.setValue(emlib.MDL_CTF_Q0,            mdGroups.getValue(emlib.MDL_CTF_Q0,idGroup),            idCTFAux)
        resolutionError= mdGroups.getValue(emlib.MDL_CTF_SAMPLING_RATE,idGroup)

        counter = 0
        minDef  = mdGroups.getValue(emlib.MDL_CTF_DEFOCUSU,idGroup)
        maxDef  = minDef
        avgDef  = minDef

        mdCTFAux.setValue(emlib.MDL_CTF_DEFOCUSU, minDef, idCTFAux)
        for idGroup in mdGroups:
            defocusU = mdGroups.getValue(emlib.MDL_CTF_DEFOCUSU,idGroup)
            mdCTFAux.setValue(emlib.MDL_CTF_DEFOCUSV, defocusU, idCTFAux);
            resolution = emlib.errorMaxFreqCTFs(mdCTFAux,pi/2.)
            if  resolution > resolutionError/ctfGroupMaxDiff:
                avgDef /=  counter
                df.cleanObjId()
                df._defocusMin.set(minDef)
                df._defocusMax.set(maxDef)
                df._defocusSum.set(df._defocusSum.get()+avgDef)
                df._size.set(counter)
                setOfDefocus.append(df)
                counter = 0
                minDef = defocusU
                avgDef = defocusU
                mdCTFAux.setValue(emlib.MDL_CTF_DEFOCUSU, defocusU, idCTFAux);
            else:
                avgDef  += defocusU
                
            maxDef = defocusU
            counter += 1
        ###
        setOfDefocus.printAll()
        ###
        writeSetOfDefocusGroups(setOfDefocus, fnXmipp)
        self._defineOutputs(outputDefocusGroups=setOfDefocus)
        
    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        """ The function of this hook is to add some validation before the protocol
        is launched to be executed. It should return a list of errors. If the list is
        empty the protocol can be executed.
        """
        errors = [ ] 
        # Add some errors if input is not valid
        return errors
    
    def _citations(self):
        cites = []            
        return cites
    
    def _summary(self):
        summary = []

        return summary
    
    def _methods(self):
        return self._summary()  # summary is quite explicit and serve as methods
    