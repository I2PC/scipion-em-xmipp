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

    AI Generated:

    # Defocus Group (XmippProtCTFDefocusGroup) — User Manual

    ## Overview

    The Defocus Group protocol divides a set of particles into groups according
    to their defocus values. Its main purpose is to organize particles into
    subsets whose CTF behavior is sufficiently similar that they can be
    treated together in later analyses.

    In practical cryo-EM workflows, defocus grouping is often used when one
    wants to study or process particles while accounting for variations in
    imaging conditions across the dataset. Since the CTF depends strongly on
    defocus, particles acquired at different defocus values may not be
    equivalent from the point of view of image formation. Grouping them helps
    preserve a better balance between experimental realism and computational
    simplicity.

    For a biological user, this protocol is not a data-cleaning step but a way
    of structuring the dataset. It is especially useful when downstream methods
    benefit from dividing particles into CTF-homogeneous subsets.

    ## Inputs and General Workflow

    The protocol requires a **set of particles with associated CTF information**.
    In particular, each particle must carry CTF metadata, since the grouping is
    based on defocus.

    The protocol examines the defocus values of the particles, sorts them, and
    then separates them into groups such that particles within the same group
    have sufficiently similar CTF behavior. The output is a set of defocus
    groups, each defined by a minimum and maximum defocus value and the number
    of particles assigned to that interval.

    Although the CTF is a multidimensional function, this protocol uses
    primarily the **defocus U** value to create the groups. In that sense, it
    is a simplified, one-dimensional grouping strategy.

    ## Why Defocus Grouping Matters

    The contrast transfer function changes with defocus, and this affects how
    structural information is represented in the images. If particles with very
    different defocus values are pooled together indiscriminately, their CTF
    modulation may differ enough to complicate averaging, classification, or
    interpretation.

    Defocus grouping addresses this by creating subsets of particles whose CTFs
    are close enough to be considered similar for practical purposes.
    Biologically, this does not change the underlying structure of the sample,
    but it improves the consistency of how that structure is represented in
    the images.

    This can be useful in workflows that model or compensate for CTF effects
    at the group level rather than for each particle independently.

    ## Grouping Criterion

    The key parameter in this protocol is the **error for grouping**. This
    parameter controls how different two defocus values are allowed to be
    before the particles are assigned to different groups.

    The criterion is based on the frequency at which the phase difference
    between the CTFs reaches approximately 90 degrees. If the difference
    between two defocus values would cause their CTF phases to diverge too
    strongly, the protocol places them in separate groups.

    Conceptually, this means that the grouping is not based on an arbitrary
    numerical defocus interval, but on a physically meaningful estimate of
    when two CTFs become too different.

    ## Interpreting the Grouping Parameter

    The grouping parameter determines how finely the dataset is divided.

    Smaller grouping tolerance leads to broader acceptance of defocus variation
    within a group, and therefore to fewer, larger groups. Larger grouping
    strictness creates more groups, each containing particles with more similar
    defocus values.

    From a practical perspective, there is a trade-off. If the groups are too
    broad, important CTF differences may be ignored. If they are too narrow,
    the dataset may become fragmented into many small groups, which can be
    inconvenient or statistically weak for downstream processing.

    In most biological workflows, the right choice depends on dataset size and
    the intended downstream use. Large datasets can often tolerate finer
    grouping, whereas smaller datasets may require more conservative grouping
    to keep enough particles per group.

    ## Simplifications and Limitations

    This protocol groups particles mainly according to **defocus U**, so it
    does not fully represent all possible differences in CTF shape. In
    particular, astigmatism and other CTF parameters are not the main drivers
    of the grouping, even though they may still influence image formation.

    This simplification is often acceptable when defocus is the dominant source
    of CTF variation, but users should keep in mind that it is an approximation.
    For datasets with strong astigmatism or unusually heterogeneous imaging
    conditions, the groups may not capture all meaningful differences.

    Therefore, the protocol should be understood as a practical and physically
    motivated grouping tool, not as a complete description of CTF heterogeneity.

    ## Outputs and Their Interpretation

    The protocol produces a **set of defocus groups**. Each group corresponds
    to a defocus interval and contains information such as:
    * the minimum defocus in the group
    * the maximum defocus in the group
    * the number of particles assigned to that group

    This output can be used to inspect how defocus is distributed across the
    dataset and to organize downstream analysis accordingly.

    Biologically, the groups do not correspond to different structural states,
    but to different imaging conditions. Their meaning is therefore technical
    rather than conformational.

    ## Practical Recommendations

    This protocol is most useful when downstream methods benefit from handling
    particles in CTF-similar subsets. It can also be valuable as an exploratory
    tool to understand the distribution of defocus values in a dataset.

    A good practice is to inspect the number and size of the resulting groups.
    If the protocol generates too many very small groups, the grouping may be
    too strict. If it generates only a few very broad groups, the grouping may
    be too permissive.

    Users should also remember that defocus grouping is not a substitute for
    proper CTF estimation or quality control. It assumes that the CTF metadata
    attached to the particles are already reliable.

    ## Final Perspective

    The Defocus Group protocol provides a practical way to partition a particle
    dataset according to the similarity of CTF conditions. By organizing
    particles into defocus-consistent subsets, it helps downstream analyses
    account for one of the main experimental sources of variation in cryo-EM
    imaging.

    For most biological users, it should be seen as a dataset organization tool
    that improves CTF consistency at the group level, rather than as a
    filtering or correction procedure.
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
    