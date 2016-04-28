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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
Since the Projection Matching protocol of Xmipp 3 has a very large
form definition, we have separated in this sub-module.
"""


from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, BooleanParam, IntParam, 
                                        FloatParam, StringParam, Positive, GE,
                                        EnumParam, NumericListParam, TextParam,
                                        DigFreqParam)
                                        

def _defineProjectionMatchingParams(self, form):
    import pyworkflow.em.packages.xmipp3 as xmipp3
    
    form.addSection(label='Input')
    
    #SelFileName
    form.addParam('inputParticles', PointerParam, label="Input particles", important=True, 
                 pointerClass='SetOfParticles', 
                 help='Select the input particles. \n '
                      'If you want perform *CTF* correction the input particles \n '
                      'should have information about the CTF (hasCTF=True)')  
    form.addParam('useInitialAngles', BooleanParam, default=False,
                 label="Use initial angles/shifts ? ", 
                 help='Set to *Yes* if you want to use the projection assignment (angles/shifts) \n '
                 'associated with the input particles (hasProjectionAssigment=True)')
    # ReferenceFileNames      
    form.addParam('input3DReferences', PointerParam,
                 pointerClass='Volume',
                 label='Initial 3D reference volume', 
                 help='Input 3D reference reconstruction.')
    form.addParam('cleanUpFiles', BooleanParam, default=False,
                 label="Clean up intermediate files?",  expertLevel=LEVEL_ADVANCED,
                 help='Save disc space by cleaning up intermediate files. \n '
                      'Be careful, many options of the visualization protocol will not work anymore, \n '
                      'since all class averages, selfiles etc will be deleted. ')
    
    form.addSection(label='CTF correction & Mask')
    
    groupCTF = form.addGroup('CTF correction')
    
    groupCTF.addParam('doCTFCorrection', BooleanParam, default=True,
                 label="Perform CTF correction?", 
                 help='If set to true, a CTF (amplitude and phase) corrected map will be refined, \n '
                      'and the data will be processed in CTF groups. \n\n '
                      '_NOTE_: You cannot combine CTF-correction with re-alignment of the classes. \n '
                      'Remember that CTF information should be provided in the images input file. \n ')    
         
    groupCTF.addParam('doAutoCTFGroup', BooleanParam, default=True, condition='doCTFCorrection',
                 label="Make CTF groups automatically?", 
                 help='Make CTF groups based on a maximum differences at a given resolution limit. \n '
                      '_NOTE_: If this option is set to false, a docfile with the defocus values where to  \n\n '
                      'split the images in distinct defocus group has to be provided (see expert option below) \n ')             
         
    groupCTF.addParam('ctfGroupMaxDiff', FloatParam, default=0.1, condition='doCTFCorrection and doAutoCTFGroup',
                 label='Maximum difference for grouping', validators=[Positive],
                 help='If the difference between the CTF-values up to the resolution limit specified \n '
                      'below is larger than the value given here, two images will be placed in \n '
                      'distinct CTF groups.')          
    
    groupCTF.addParam('ctfGroupMaxResol', FloatParam, default=5.6, condition='doCTFCorrection and doAutoCTFGroup',
                 label='Resolution limit (A) for grouping', validators=[Positive],
                 help='Maximum resolution where to consider CTF-differences among different groups. \n '
                      'One should use somewhat higher resolutions than those aimed for in the refinement.')       
    
    # SplitDefocusDocFile
    groupCTF.addParam('setOfDefocus', StringParam,
                 label='Set of defocus', default='', condition='doCTFCorrection and not doAutoCTFGroup',
                 help='Set with defocus values where to split into groups. \n '
                      'This field is compulsory if you do not want to make the CTF groups automatically. \n\n '
                      '_NOTE_: The requested docfile can be made initially with the *xmipp_ctf_group* program, \n '
                      'and then it can be edited manually to suit your needs.')
    
    groupCTF.addParam('paddingFactor', FloatParam, default=2, condition='doCTFCorrection',
                 label='Padding factor', validators=[GE(1)],
                 help='Application of CTFs to reference projections and of Wiener filter \n '
                       'to class averages will be done using padded images. \n '
                       'Use values larger than one to pad the images.')        
    
    groupCTF.addParam('wienerConstant', FloatParam, default=-1, condition='doCTFCorrection',
                 label='Wiener constant',  expertLevel=LEVEL_ADVANCED,
                 help='Term that will be added to the denominator of the Wiener filter. \n '
                       'In theory, this value is the inverse of the signal-to-noise ratio \n '
                       'If a negative value is taken, the program will use a default value as in FREALIGN \n '
                       '(i.e. 10% of average sum terms over entire space)  \n '
                       'see Grigorieff JSB 157 (2006) pp117-125')   
    
    #TODO: Use common mask parameters
    
    groupMask = form.addGroup('Mask')
    
    # doMask, doSphericalMask now merged into maskType
    
    groupMask.addParam('maskType', EnumParam, choices=['None', 'circular', 'mask object'], default=xmipp3.MASK2D_CIRCULAR, 
                 label="Mask reference volumes", display=EnumParam.DISPLAY_COMBO,
                 help='Masking the reference volume will increase the signal to noise ratio. \n '
                      'Do not provide a very tight mask. \n ')
    
    groupMask.addParam('maskRadius', IntParam, default=-1, condition='maskType == 1',
                 label='Radius of spherical mask (px)',
                 help='This is the radius (in pixels) of the spherical mask ')       
    
    groupMask.addParam('inputMask', PointerParam, pointerClass="VolumeMask", allowNull=True, 
                 label='Mask Object', condition='maskType == 2',
                 help='The mask file should have the same dimensions as your input particles. \n '
                      'The protein region should be 1 and the solvent should be 0.')  
    
    # DataArePhaseFlipped , now taken from inputParticles.isPhaseFlipped()
    # ReferenceIsCtfCorrected, now taken from input3DReferences.isAmplitudeCorrected()
    
    form.addSection(label='Projection Matching')
    
    form.addParam('numberOfIterations', IntParam, default=4,
             label='Number of iterations',
             help='Number of iterations to perform.')
    form.addParam('innerRadius', NumericListParam, default='0', 
                 label='Inner radius for rotational correlation:', 
                 help=""" In pixels from the image center
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "8 8 2 2 " 
    specifies 4 iterations, the first two set the value to 8 
    and the last two to 2. An alternative compact notation 
    is ("2x8 2x0", i.e.,
    2 iterations with value 8, and 2 with value 2).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)
         
    form.addParam('outerRadius', NumericListParam, default='64', 
                 label='Outer radius for rotational correlation', 
                 help=""" In pixels from the image center. Use a negative number to use the entire image.
    *WARNING*: this radius will be use for masking before computing resolution
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "8 8 2 2 " 
    specifies 4 iterations, the first two set the value to 8 
    and the last two to 2. An alternative compact notation 
    is ("2x8 2x0", i.e.,
    2 iterations with value 8, and 2 with value 2).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)        
    
    form.addParam('availableMemory', IntParam, default=2, expertLevel=LEVEL_ADVANCED, 
                 label='Available memory to store all references (Gb)',
                 help=""" This is only for the storage of the references. If your projections do not fit in memory, 
    the projection matching program will run MUCH slower. But, keep in mind that probably 
    some additional memory is needed for the operating system etc.
    Note that the memory per computing node needs to be given. That is, when using threads, 
    this value will be multiplied automatically by the number of (shared-memory) threads.
    """)
    
    form.addParam('angSamplingRateDeg', NumericListParam, default='7 5 3 2', 
                 label='Angular sampling rate (deg)',
                 help=""" Angular distance (in degrees) between neighboring projection  points
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "8 8 2 2 " 
    specifies 4 iterations, the first two set the value to 8 
    and the last two to 2. An alternative compact notation 
    is ("2x8 2x0", i.e.,
    2 iterations with value 8, and 2 with value 2).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)
    
    form.addParam('maxChangeInAngles', NumericListParam, default='1000 10 6 4', 
                  label='Angular search range (deg)',
                  help=""" Maximum change in rot & tilt  (in +/- degrees)
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "1000 1000 10 10 " 
    specifies 4 iterations, the first two set the value to 1000 (no restriction)
    and the last two to 10degrees. An alternative compact notation 
    is ("2x1000 2x10", i.e.,
    2 iterations with value 1000, and 2 with value 10).
    <Note:> if there are less values than iterations the last value is reused
    <Note:> if there are more values than iterations the extra value are ignored
    """)        
    
    form.addParam('perturbProjectionDirections', NumericListParam, default='0', 
                 label='Perturb projection directions?', expertLevel=LEVEL_ADVANCED,
                 help=""" If set to 1, this option will result to a Gaussian perturbation to the 
    evenly sampled projection directions of the reference library. 
    This may serve to decrease the effects of model bias.
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "1 1 0" 
    specifies 3 iterations, the first two set the value to 1 
    and the last to 0. An alternative compact notation 
    is ("2x1 0", i.e.,
    2 iterations with value 1, and 1 with value 0).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)   
    
    # Changed from String to Int 
    form.addParam('projectionMethod', EnumParam, choices=['fourier', 'real_space'], 
                 default=xmipp3.PROJECT_REALSPACE, expertLevel=LEVEL_ADVANCED, 
                 label="Projection method", display=EnumParam.DISPLAY_COMBO,
                 help='select projection method, by default Fourier with padding 1 and interpolation bspline')        
    
    
    form.addParam('paddingAngularProjection', FloatParam, default=1, expertLevel=LEVEL_ADVANCED,  
                 condition='projectionMethod == %d' % xmipp3.PROJECT_FOURIER,
                 label='Padding factor for projection', validators=[GE(1)],
                 help="""Increase the padding factor will improve projection quality but 
    projection generation will be slower. In general padding 1 and spline is OK
    """)       
    # Changed from String to Int 
    form.addParam('kernelAngularProjection', EnumParam, choices=['neareast', 'linear', 'bspline'],
                 default=xmipp3.KERNEL_BSPLINE, expertLevel=LEVEL_ADVANCED,  
                 condition='projectionMethod == %d' % xmipp3.PROJECT_FOURIER,
                 label='Interpolation kernel for projection', 
                 help=""" Interpolation kernel for the generation of projections.
    """)
    
    form.addParam('maxChangeOffset', NumericListParam, default='1000 10 5', 
                 label='Maximum change in origin offset', expertLevel=LEVEL_ADVANCED,
                 help=""" Maximum shift allowed per iteration.
    You may specify this option for each iteration.
    This can be done by a sequence of numbers (for instance, "1000 10 5"
    specifies 3 iterations, the first two set the value to 1000
    (almost no restriction) and the last to 5.
    An alternative compact notation
    is ("2x1000 5", i.e.,
    2 iterations with value 1000, and 1 with value 5).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra values are ignored
    """)          
    
    form.addParam('search5DShift', NumericListParam, default='4x5 0', 
                 label='Search range for 5D translational search',
                 help=""" Give search range from the image center for 5D searches (in +/- pixels).
    Values larger than 0 will results in 5D searches (which may be CPU-intensive)
    Give 0 for conventional 3D+2D searches. 
    Note that after the 5D search, for the optimal angles always 
    a 2D exhaustive search is performed anyway (making it ~5D+2D)
    Provide a sequence of numbers (for instance, "5 5 3 0" specifies 4 iterations,
    the first two set the value to 5, then one with 3, resp 0 pixels.
    An alternative compact notation is ("3x5 2x3 0", i.e.,
    3 iterations with value 5, and 2 with value 3 and the rest with 0).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    
    """)  
    form.addParam('search5DStep', NumericListParam, default='2', 
                 label='Step size for 5D translational search', expertLevel=LEVEL_ADVANCED,
                 help="""" Provide a sequence of numbers (for instance, "2 2 1 1" specifies 4 iterations,
    the first two set the value to 2, then two with 1 pixel.
    An alternative compact notation is ("2x2 2x1", i.e.,
    2 iterations with value 2, and 2 with value 1).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)          
    
    form.addParam('doRestricSearchbyTiltAngle', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                 label="Restrict tilt angle search?", 
                 help ='Restrict tilt angle search \n ')             
    
    form.addParam('tilt0', FloatParam, default=0, condition='doRestricSearchbyTiltAngle',
                 label="Lower-value for restricted tilt angle search", 
                 help ='Lower-value for restricted tilt angle search \n ')             
    
    form.addParam('tiltF', FloatParam, default=180, condition='doRestricSearchbyTiltAngle',
                 label="Higher-value for restricted tilt angle search", 
                 help ='Higher-value for restricted tilt angle search \n ')             
    form.addParam('symmetry', TextParam, default='c1',
                 label='Point group symmetry',
                 help=""" See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]
    for a description of the symmetry groups format
    If no symmetry is present, give c1
    """)
    form.addParam('symmetryGroupNeighbourhood', TextParam, default='', expertLevel=LEVEL_ADVANCED,
                 label='Symmetry group for Neighbourhood computations',
                 help=""" If you do not know what this is leave it blank.
    This symmetry will be using for compute neighboring points,
    but not for sampling or reconstruction
    See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]
    for a description of the symmetry groups format
    If no symmetry is present, give c1
    """
    )
    form.addParam('onlyWinner', NumericListParam, default='0', 
                 label='compute only closest neighbor', expertLevel=LEVEL_ADVANCED,
                 condition="symmetryGroupNeighbourhood != ''",
                 help="""This option is only relevant if symmetryGroupNeighbourhood !=''
    If set to 1 only one neighbor will be computed per sampling point
    You may specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "1 1 0" 
    specifies 3 iterations, the first two set the value to 1 
    and the last to 0. An alternative compact notation 
    is ("2x1 0", i.e.,
    2 iterations with value 1, and 1 with value 0).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)     
    
    form.addParam('discardImages', EnumParam, 
                 choices=['None', 'maxCC', 'percentage', 'classPercentage'],
                 default=xmipp3.SELECT_NONE, display=EnumParam.DISPLAY_COMBO,
                 label='Discard images?', 
                 help=""" 
    None : No images will be discarded.
    maxCC  : Minimum Cross Correlation, discard images with CC below a fixed value.
    percentage : Discard percentage of images with less CC.
    classPercentage: Discard percentage of images in each projection direction with less CC.
    Value of each option is set below.
    """)
    form.addParam('minimumCrossCorrelation', NumericListParam, default='0.1', 
                 label='discard image if CC below', 
                 condition='discardImages==%d' % xmipp3.SELECT_MAXCC,
                 help=""" 
    Discard images with cross-correlation (CC) below this value.
    Provide a sequence of numbers (for instance, "0.3 0.3 0.5 0.5" specifies 4 iterations,
    the first two set the value to 0.3, then two with 0.5.
    An alternative compact notation would be ("2x0.3 2x0.5").
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)
    
    form.addParam('discardPercentage', NumericListParam, default='10', 
                 label='discard image percentage with less CC',
                 condition='discardImages==%d'%xmipp3.SELECT_PERCENTAGE,
                 help=""" 
    Discard this percentage of images with less cross-correlation (CC)
    Provide a sequence of numbers (for instance, "20 20 10 10" specifies 4 iterations,
    the first two set the value to 20%, then two with 10%
    An alternative compact notation would be ("2x20 2x10").
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    Set to zero to prevent discarding any images
    """)
    
    form.addParam('discardPercentagePerClass', NumericListParam, default='10', 
                 label='discard image percentage in class with less CC',
                 condition='discardImages==%d'%xmipp3.SELECT_CLASSPERCENTAGE,
                 help=""" 
    Discard this percentage of images in each class(projection direction)
    with less cross-correlation (CC)    
    Provide a sequence of numbers (for instance, "20 20 10 10" specifies 4 iterations,
    the first two set the value to 20%, then two with 10%
    An alternative compact notation would be ("2x20 2x10").
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    Set to zero to prevent discarding any images
    """)
    
    form.addParam('doScale', BooleanParam, default=False,
                 label="Perform scale search?",  expertLevel=LEVEL_ADVANCED,
                 help=' If true perform scale refinement. (UNDER DEVELOPMENT!!!!) \n  ')
    
    form.addParam('scaleStep', NumericListParam, default=1, condition='doScale',
                 label='Step scale factors size',
                 help='''Scale step factor size (1 means 0.01 in/de-crements around 1).
    Provide a sequence of numbers (for instance, "1 1 .5 .5" specifies 4 iterations,
    the first two set the value to 1%, then two with .5%
    An alternative compact notation would be ("2x1 2x0.5").
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    Set to zero to prevent discarding any images''')  
    
    form.addParam('scaleNumberOfSteps', NumericListParam, default=3, condition='doScale',
                 label='Number of scale steps',
                 help=""" 
    Number of scale steps.
    With default values (ScaleStep='1' and ScaleNumberOfSteps='3'): 1 +/-0.01 | +/-0.02 | +/-0.03.    
    With values ScaleStep='2' and ScaleNumberOfSteps='4' it performs a scale search over:
    1 +/-0.02 | +/-0.04 | +/-0.06 | +/-0.08.    
    In general scale correction should only be applied to the last iteration. Do not use it unless
    your data is fairly well aligned.
    """)  
    
    form.addParam('projMatchingExtra', StringParam, default='',
                 label='Additional options for Projection_Matching', expertLevel=LEVEL_ADVANCED,
                 help=""" For details see:
    [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Projection_matching][projection matching]] and
    [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Mpi_projection_matching][mpi projection matching]]
    try -Ri xx -Ro yy for restricting angular search (xx and yy are
    the particle inner and outter radius)
    """)
    #DoSaveImagesAssignedToClasses    you can get this information in visualize
    form.addSection(label='2D re-alignment of classes', expertLevel=LEVEL_ADVANCED)
    
    form.addParam('performAlign2D', BooleanParam, default=False,
                 label='Perform 2D re-alignment')
    
    form.addParam('doAlign2D', NumericListParam, default='0', condition='performAlign2D',
                 label='Perform 2D re-alignment of classes?',
                 help=""" After performing a 3D projection matching iteration, each of the
    subsets of images assigned to one of the library projections is
    re-aligned using a 2D-alignment protocol.
    This may serve to remove model bias.
    For details see:
    [[http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Align2d][align 2d]]
    Note that you cannot combine this option with CTF-correction!
    You may specify this option for each iteration. 
    This can be done by a sequence of 0 or 1 numbers (for instance, "1 1 0 0" 
    specifies 4 iterations, the first two applied alig2d while the last 2
    dont. An alternative compact notation is 
    is ("2x1 2x0", i.e.,
    2 iterations with value 1, and 2 with value 0).
    *Note:*if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    *IMPORTANT:* if you set this variable to 0 the output  of the projection
    muching step will be copied as output of align2d
    """)
    
    form.addParam('align2DIterNr', NumericListParam, default='4', condition='performAlign2D',
                 label='Number of align2d iterations:', expertLevel=LEVEL_ADVANCED,
                 help=""" Use at least 3 iterations
    The number of align iteration may change in each projection matching iteration
    Ffor instance, "4 4 3 3 " 
    specifies 4 alig2d iterations in the first projection matching iteration 
    and  two 3 alig2d iteration in the last 2 projection matching iterations.
    An alternative compact notation 
    is ("2x4 2x3", i.e.,
    2 iterations with value 4, and 2 with value 3).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)        
    
    
    form.addParam('align2dMaxChangeOffset', NumericListParam, default='2x1000 2x10', 
                 condition='performAlign2D',
                 label='Maximum change in origin offset (+/- pixels)', expertLevel=LEVEL_ADVANCED,
                 help="""Maximum change in shift  (+/- pixels)
    You must specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "1000 1000 10 10 " 
    specifies 4 iterations, the first two set the value to 1000 (no restriction)
    and the last two to 10degrees. An alternative compact notation 
    is ("2x1000 2x10", i.e.,
    2 iterations with value 1000, and 2 with value 10).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)    
    
    form.addParam('align2dMaxChangeRot', NumericListParam, default='2x1000 2x20', 
                 condition='performAlign2D',
                 label='Maximum change in rotation (+/- degrees)', expertLevel=LEVEL_ADVANCED,
                 help="""Maximum change in shift  (+/- pixels)
    You must specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, "1000 1000 10 10 " 
    specifies 4 iterations, the first two set the value to 1000 (no restriction)
    and the last two to 10degrees. An alternative compact notation 
    is ("2x1000 2x10", i.e.,
    2 iterations with value 1000, and 2 with value 10).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)     
    
    
    form.addSection(label='3D Reconstruction')
    
    form.addParam('reconstructionMethod', EnumParam, expertLevel=LEVEL_ADVANCED,
                 choices=['fourier', 'art', 'wbp'],
                 default=xmipp3.RECONSTRUCT_FOURIER, display=EnumParam.DISPLAY_COMBO,
                 label='Reconstruction method', 
                 help=""" Select what reconstruction method to use.
    fourier: Fourier space interpolation (with griding).
    art: Agebraic reconstruction technique
    wbp : Weight back project method.
    """)
    
    form.addParam('fourierMaxFrequencyOfInterest', DigFreqParam, default=0.25,
                 condition='reconstructionMethod == %d' % xmipp3.RECONSTRUCT_FOURIER,
                 label='Initial maximum frequency', expertLevel=LEVEL_ADVANCED,
                 help=""" This number is only used in the first iteration. 
    From then on, it will be set to resolution computed in the resolution section
    """)         
    form.addParam('fourierReconstructionExtraCommand', StringParam, default='',
                 condition='reconstructionMethod == %d' % xmipp3.RECONSTRUCT_FOURIER,
                 label='Additional parameters for fourier', expertLevel=LEVEL_ADVANCED,
                 help=""" For details see:
    [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Fourier][fourier]]
    """)          
    
    form.addParam('artLambda', NumericListParam, default='0.2', 
                 condition='reconstructionMethod == %d' % xmipp3.RECONSTRUCT_ART,
                 label='Values of lambda for ART', expertLevel=LEVEL_ADVANCED,
                 help=""" *IMPORTANT:* ou must specify a value of lambda for each iteration even
    if ART has not been selected.
    *IMPORTANT:* NOte that we are using the WLS version of ART that 
    uses geater lambdas than the plain art.
    See for details:
    [[http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Art][xmipp art]]
    You must specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, ".1 .1 .3 .3" 
    specifies 4 iterations, the first two set the value to 0.1 
    (no restriction)
    and the last  two to .3. An alternative compact notation 
    is ("2x.1 2x.3").
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)   
    
    form.addParam('artReconstructionExtraCommand', StringParam, default='-k 0.5 -n 10 ',
                 condition='reconstructionMethod == %d' % xmipp3.RECONSTRUCT_ART,
                 label='Additional parameters for ART', expertLevel=LEVEL_ADVANCED,
                 help=""" For details see:
    [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Art][xmipp art]]
    """)          
    
    form.addParam('wbpReconstructionExtraCommand', StringParam, default='',
                 condition='reconstructionMethod == %d' % xmipp3.RECONSTRUCT_WBP,
                 label='Additional parameters for WBP', expertLevel=LEVEL_ADVANCED,
                 help=""" For details see:
    [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Wbp][xmipp wbp]]
    """)                  
    
    form.addParam('doComputeResolution', BooleanParam, default=True,
                 label='Compute resolution?', expertLevel=LEVEL_ADVANCED,
                 help=""" For details see:
    [[http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Resolution][xmipp resolution]].
    """)
    
    form.addParam('doSplitReferenceImages', NumericListParam, default='1',
                 label='Split references averages?', expertLevel=LEVEL_ADVANCED,
                 condition="doComputeResolution",
                 help="""In theory each reference average should be splited
    in two when computing the resolution. In this way each
    projection direction will be represented in each of the
    subvolumes used to compute the resolution. A much faster
    but less accurate approach is to split the 
    proyection directions in two but not the averages. We
    recommend the first approach for small volumes and the second for
    large volumes (especially when using small angular
    sampling rates.
    *IMPORTANT:* the second option has ONLY been implemented for FOURIER
    reconstruction method. Other reconstruction methods require this
    flag to be set to True
    You may specify this option for each iteration. 
    This can be done by a sequence of 0 or 1 numbers (for instance, "1 1 0 0" 
    specifies 4 iterations, the first two split the images   while the last 2
    don't. an alternative compact notation is 
    is ("2x1 2x0", i.e.,
    2 iterations with value 1, and 2 with value 0).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more vapplications/scripts/protocols/new_protocol_projmatch.pyalues than iterations the extra value are ignored
    """)            
    
    form.addParam('doLowPassFilter', BooleanParam, default=True,
                 label="Low-pass filter the reference?")
    
    form.addParam('useFscForFilter', BooleanParam, default=True,
                  label='Use estimated resolution for low-pass filtering?',
                  condition="doLowPassFilter",
                  help=""" If set to true, the volume will be filtered at a frecuency equal to
   the  resolution computed with a FSC=0.5 threshold, possibly 
   plus a constant provided by the user in the next input box. 

   If set to false, then the filtration will be made at the constant 
   value provided by the user in the next box (in digital frequency, 
   i.e. pixel^-1: minimum 0, maximum 0.5)
    """)
    
    form.addParam('constantToAddToFiltration', NumericListParam, default='0.05',
                 label='Constant to be added to the estimated resolution',
                 condition="doLowPassFilter",
                 help=""" The meaning of this field depends on the previous flag.
    If set to true, then the volume will be filtered at a frequency equal to
    the  resolution computed with resolution_fsc (FSC=0.5) plus the value 
    provided in this field 
    If set to false, the volume will be filtered at the resolution
    provided in this field 
    This value is in digital frequency, or pixel^-1: minimum 0, maximum 0.5
    
    If you detect correlation between noisy regions decrease this value 
    (even to negative values)
    
    You can specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, ".15 .15 .1 .1"
    specifies 4 iterations, the first two set the constant to .15
    and the last two to 0.1. An alternative compact notation 
    is ("2x.15 2x0.1", i.e.,
    4 iterations with value 0.15, and three with value .1).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)

    form.addParam('constantToAddToMaxReconstructionFrequency', NumericListParam, default='0.1',
                 label='Constant to be added to the reconstruction maximum frequency', expertLevel=LEVEL_ADVANCED,
                 condition="doLowPassFilter",
                 help=""" The meaning of this field depends on the <use FSC for filter> flag.
    If set to true, then the volume will be reconstructed up to the frequency equal to
    the resolution computed with resolution_fsc (FSC=0.5) plus the value 
    provided in this field 
    If set to false, the volume will be reconstructed up to the resolution
    provided in this field 
    This value is in digital frequency, or pixel^-1: minimum 0, maximum 0.5
    
    You can specify this option for each iteration. 
    This can be done by a sequence of numbers (for instance, ".15 .15 .1 .1" 
    specifies 4 iterations, the first two set the constant to .15
    and the last two to 0.1. An alternative compact notation 
    is ("2x.15 2x0.1", i.e.,
    4 iterations with value 0.15, and three with value .1).
    *Note:* if there are less values than iterations the last value is reused
    *Note:* if there are more values than iterations the extra value are ignored
    """)
    
    form.addParam('mpiJobSize', IntParam, default=2,
                  label='MPI job size', expertLevel=LEVEL_ADVANCED,
                  help="""Minimum size of jobs in mpi processes.
    Set to 1 for large images (e.g. 500x500)
    and to 10 for small images (e.g. 100x100)
    """)

    form.addParallelSection(threads=1, mpi=8)
        
