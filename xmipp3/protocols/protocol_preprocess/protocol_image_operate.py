# *****************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), Sep 2013
# * Ported to Scipion:
# *           Vahid Abrishami (vabrishami@cnb.csic.es), Oct 2014
# *
# * Refactored/Updated: Josue Gomez-Blanco (josue.gomez-blanco@mcgill.ca), Jun 2016
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
# *****************************************************************************

from collections import OrderedDict

from pwem.constants import ALIGN_NONE
import pyworkflow.protocol.params as params
from pwem.protocols import ProtOperateParticles, ProtOperateVolumes

from xmipp3.convert import (writeSetOfParticles, writeSetOfVolumes,
                            getImageLocation)
from .protocol_process import XmippProcessParticles, XmippProcessVolumes


# Operands enum
OP_PLUS = 0
OP_MINUS = 1
OP_MULTIPLY = 2
OP_DIVIDE = 3
OP_MINIMUM = 4
OP_MAXIMUM = 5
OP_DOTPRODUCT = 6
OP_LOG = 7
OP_LOG10 = 8
OP_SQRT = 9
OP_ABS = 10
OP_POW = 11
OP_SLICE = 12
OP_COLUNM = 13
OP_ROW = 14
OP_RADIAL = 15
OP_RESET = 16

OP_CHOICES = OrderedDict() #[-1]*(OP_RESET+1)

OP_CHOICES[OP_PLUS]  = 'plus'
OP_CHOICES[OP_MINUS] = 'minus'
OP_CHOICES[OP_MULTIPLY] = 'multiply'
OP_CHOICES[OP_DIVIDE] = 'divide'
OP_CHOICES[OP_MINIMUM] = 'minimum'
OP_CHOICES[OP_MAXIMUM] = 'maximum'
OP_CHOICES[OP_DOTPRODUCT]  = 'dot product'
OP_CHOICES[OP_LOG] = 'log'
OP_CHOICES[OP_LOG10] = 'log10'
OP_CHOICES[OP_SQRT] = 'sqrt'
OP_CHOICES[OP_ABS] = 'abs'
OP_CHOICES[OP_POW] = 'pow'
OP_CHOICES[OP_COLUNM] = 'colunm'
OP_CHOICES[OP_SLICE] = 'slice'
OP_CHOICES[OP_ROW] = 'row'
OP_CHOICES[OP_RADIAL] = 'radial average'
OP_CHOICES[OP_RESET] = 'reset'


binaryCondition = ('(operation == %d or operation == %d or operation == %d or '
                   'operation == %d or operation == %d or operation == %d) ' %
                   (OP_PLUS, OP_MINUS, OP_MULTIPLY,
                    OP_DIVIDE, OP_MINIMUM, OP_MAXIMUM))

#noValueCondition = '(operation == 7 or operation == 8 or operation == 9 or '\
#                   'operation == 10 or operation == 15 or operation == 16) '
noValueCondition = '(operation == %d or operation == %d or operation == %d or '\
                   'operation == %d or operation == %d or operation == %d) '%\
                   (OP_LOG, OP_LOG10, OP_SQRT,\
                    OP_ABS, OP_POW, OP_RESET)

intValueCondition = '(operation == %d or operation == %d)'%(OP_COLUNM, OP_ROW)

dotCondition = 'operation == %d'%OP_DOTPRODUCT
powCondition = 'operation == %d'%OP_POW

operationDict = {OP_PLUS : ' --plus ', OP_MINUS : ' --minus ',
                 OP_MULTIPLY : ' --mult ', OP_DIVIDE : ' --divide ',
                 OP_MINIMUM : ' --min ', OP_MAXIMUM : ' --max ',
                 OP_DOTPRODUCT : ' --dot_product ', OP_LOG : ' --log ',
                 OP_LOG10 : ' --log10', OP_SQRT : ' --sqrt ',
                 OP_ABS : ' --abs ', OP_POW : ' --pow ',
                 OP_SLICE : ' --slice ',  OP_RADIAL : ' --radial_avg ',
                 OP_RESET : ' --reset ', OP_COLUNM: '--column',
                 OP_ROW: '--row'}


class XmippOperateHelper():
    """ Some image operations such as: Dot product or Summation. """
    _label = 'image operate'

    def __init__(self, **args):
        self._program = "xmipp_image_operate"

    #--------------------------- DEFINE param functions -----------------------
    def _defineProcessParams(self, form):
        form.addParam('operation', params.EnumParam, choices=list(OP_CHOICES.values()),
                      default=OP_PLUS,
                      label="Operation",
                      help="Binary operations: \n"
                           "*plus*: Sums two images, volumes or adds a "
                           "numerical value to an image. \n"
                           "*minus*: Subtracts two images, volumes or "
                           "subtracts a numerical value to an image. \n"
                           "*multiply*: Multiplies two images, volumes, or "
                           "multiplies per a given number. \n"
                           "*divide*: Divides two images, volumes, or divides "
                           "per a given number. \n"
                           "*minimum*: Minimum of two images, volumes, or "
                           "number (pixel-wise). \n"
                           "*maximum*: Maximum of two images, volumes, or "
                           "number (pixel-wise). \n"
                           "*dot product*: Dot product between two images or"
                           " volumes. \n"
                           "Unary operations: \n"
                           "*log*: Computes the natural logarithm of an "
                           "image. \n"
                           "*log10*: Computes the decimal logarithm of "
                           "an image. \n"
                           "*sqrt*: Computes the square root of an image \n"
                           "*abs*: Computes the absolute value of an image. \n"
                           "*pow*: Computes the power of an image. \n"
                           "*slice*: Extracts a given slice from a volume "
                           "(first slice=0). \n"
                           "*column*: Extracts a given column from a image "
                           "or volume. \n"
                           "*row*: Extracts a given row from a image or "
                           "volume. \n"
                           "*radial average*: Compute the radial average of "
                           "an image. \n"
                           "*reset*: Set the image to 0")
        form.addParam('isValue', params.BooleanParam, default=False,
                      label="Second operand is a value?",
                      condition=binaryCondition,
                      help="Set to true if you want to use a "
                           "value of the second operand")
        self._defineSpecificParams(form)
        form.addParam('value', params.FloatParam,
                      allowNull=True,
                      condition='isValue and %s or %s' %
                                (binaryCondition, powCondition),
                      label='Input value ',
                      help = 'Set the desire float value')
        form.addParam('intValue', params.IntParam,
                      allowNull=True,
                      condition=intValueCondition,
                      label='Input value ',
                      help = 'This value must be integer')
    
    #--------------------------- INSERT STEPS functions ------------------------
    def _insertProcessStep(self):
        operationStr = operationDict[self.operation.get()]
        self._insertFunctionStep("operationStep", operationStr)
    
    #--------------------------- UTILS functions ------------------------------
    def _isBinaryCond(self):
        operation = self.operation.get()
        return (operation == OP_PLUS or operation == OP_MINUS or
                operation == OP_MULTIPLY or operation == OP_DIVIDE
                or operation == OP_MINIMUM or operation == OP_MAXIMUM)
    
    def _isNoValueCond(self):
        operation = self.operation.get()
        return (operation == OP_LOG or operation == OP_LOG10 or
                operation == OP_SQRT or operation == OP_ABS or
                operation == OP_RADIAL or operation == OP_RESET)
    
    def _isPowCond(self):
        operation = self.operation.get()
        return operation == OP_POW
    
    def _isDotCond(self):
        operation = self.operation.get()
        return operation == OP_DOTPRODUCT
        
    
    def _isintValueCond(self):
        operation = self.operation.get()
        return (operation == OP_SLICE or operation == OP_COLUNM
                or operation == OP_ROW)
    
    def _getSecondSetFn(self):
        return self._getTmpPath("images_to_apply.xmd")


class XmippProtImageOperateParticles(ProtOperateParticles,
                                     XmippProcessParticles,
                                     XmippOperateHelper):
    """Performs operations between two sets of particles or to a set of
    particles. The mathematical operations are: plus, minus multiply, divide,
    minimum, maximum, dot product, log, log10, sqrt, abs, pow, column, slice,
    row, mn, radial average or reset. This protocol supports comparative
    analysis or creation of difference maps between conditions.

    AI Generated

    ## Overview

    The Operate Particles protocol applies mathematical operations to a set of
    particle images.

    The operation can be applied in two main ways:

    - between one particle set and a numerical value;
    - between two particle sets.

    The protocol supports arithmetic operations such as addition, subtraction,
    multiplication, division, minimum, and maximum. It also supports unary
    operations such as logarithm, square root, absolute value, power, reset,
    slice/row/column extraction, radial average, and dot product.

    The main output is a new particle set whose images contain the result of the
    selected operation.

    ## Inputs and General Workflow

    The main input is a set of particles.

    If the selected operation requires a second image operand, the user provides a
    second particle set. If the operation uses a numerical value, the user enables
    the value option and provides the number.

    The protocol converts the input particle set or sets to Xmipp metadata format,
    runs the Xmipp image-operation program, writes the resulting images to a new
    stack, and creates an output particle set that preserves the input metadata
    when possible.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be operated on.

    These particles are the first operand of the operation. For unary operations,
    this is the only image input. For binary operations, this set is combined with
    either a numerical value or a second particle set.

    The protocol writes the input particles without alignment information for the
    image-operation step, because the operation acts directly on image values.

    ## Operation

    The **Operation** parameter selects the mathematical operation to apply.

    The available operations include:

    - plus;
    - minus;
    - multiply;
    - divide;
    - minimum;
    - maximum;
    - dot product;
    - log;
    - log10;
    - sqrt;
    - abs;
    - pow;
    - slice;
    - column;
    - row;
    - radial average;
    - reset.

    Some operations require a second operand, some require a numerical value, and
    some act directly on the input images.

    ## Binary Operations

    The binary operations are:

    - plus;
    - minus;
    - multiply;
    - divide;
    - minimum;
    - maximum.

    These operations can be applied either with a numerical value or with a second
    particle set.

    For example, multiplying by a value scales every particle image by that value.
    Subtracting a second particle set subtracts corresponding particle images
    pixel by pixel.

    ## Second Operand as a Value

    The **Second operand is a value?** option is used for binary operations.

    If enabled, the second operand is the numerical **Input value** provided by the
    user.

    For example:

    - plus 1 adds 1 to all pixels;
    - multiply 0.5 scales all particles by 0.5;
    - divide 2 divides all pixel values by 2.

    This is useful for simple intensity scaling or offset operations.

    ## Second Particle Set

    The **Input Particles (2nd)** parameter is used when a binary operation or dot
    product requires a second image operand.

    The second particle set must be compatible with the first one. For binary
    operations, the two sets must have the same number of particles and the same
    image dimensions. The protocol validates these requirements.

    The operation is applied pairwise between corresponding particles.

    ## Minimum and Maximum

    The **minimum** and **maximum** operations compute pixel-wise comparisons.

    With a numerical value, each pixel is compared with the value. With a second
    particle set, each pixel is compared with the corresponding pixel in the
    matching particle image.

    These operations can be useful for clipping intensities or combining two image
    sets by selecting lower or higher values.

    ## Dot Product

    The **dot product** operation computes the dot product between corresponding
    images from two particle sets.

    This operation requires a second particle set. The two sets must have
    compatible image dimensions.

    The result can be useful for similarity measurements or technical workflows
    that require image-wise scalar products.

    ## Logarithm Operations

    The **log** operation computes the natural logarithm of the image values.

    The **log10** operation computes the decimal logarithm.

    These operations should be used only when the image values are valid for the
    logarithm. Zero or negative values may produce undefined or invalid results.

    ## Square Root and Absolute Value

    The **sqrt** operation computes the square root of the image values.

    The **abs** operation computes the absolute value.

    Square root should be used carefully if images contain negative values.
    Absolute value can be useful when the user wants to remove sign information
    from the image intensities.

    ## Power Operation

    The **pow** operation raises image values to the user-provided power.

    The **Input value** parameter defines the exponent.

    For example, a power of 2 squares the pixel values.

    This operation can strongly change image contrast and should be used with
    care, especially for negative values or non-integer exponents.

    ## Slice, Column, and Row

    The **slice**, **column**, and **row** operations extract a selected index from
    the image data.

    The **Input value** parameter for these operations is an integer.

    These operations are mostly useful for technical inspection or conversion
    workflows rather than standard particle processing.

    ## Radial Average

    The **radial average** operation computes the radial average of an image.

    This can be useful for reducing a two-dimensional image to a radial profile or
    for analyzing radially symmetric behavior.

    The operation is applied to the input particle images.

    ## Reset

    The **reset** operation sets the image to zero.

    This is useful for creating blank images with the same metadata structure as
    the input set or for technical workflows where the image content must be
    cleared.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the result of the selected image operation. The particle
    set preserves input metadata columns when possible and points to the new image
    stack generated by Xmipp.

    The output can be used in later Scipion protocols like any other particle set.

    ## Validation Rules

    When a binary operation uses a second particle set, the two particle sets must
    have the same number of particles and the same image dimensions.

    When dot product is selected, the two particle sets must have compatible image
    dimensions.

    If these conditions are not met, the protocol reports validation errors.

    ## Interpreting the Result

    The output should be interpreted as a direct mathematical transformation of the
    input particle images.

    The protocol does not perform alignment, classification, reconstruction,
    filtering, or biological validation. It only changes image values according to
    the selected operation.

    For this reason, the operation should be chosen with a clear processing goal.

    ## Practical Recommendations

    Use value-based operations for simple intensity scaling or offset correction.

    Use binary operations between two particle sets only when the sets correspond
    particle by particle and have the same dimensions.

    Use logarithm, square root, and power operations cautiously, checking that the
    input intensity range is appropriate.

    Inspect a representative subset of output particles before using the result in
    downstream processing.

    Keep the original particle set unchanged so that the operation can be repeated
    or adjusted if needed.

    ## Final Perspective

    Operate Particles is a general image-arithmetic utility for particle sets.

    For biological users, its value is practical: it allows direct mathematical
    manipulation of particle images for preprocessing, comparison, simulation,
    debugging, or specialized workflows.

    The protocol should be understood as a low-level image-operation tool. Its
    output is only as meaningful as the operation selected by the user.
    """
    _label = 'operate particles'
    
    def __init__(self, **args):
        ProtOperateParticles.__init__(self, **args)
        XmippProcessParticles.__init__(self)
        XmippOperateHelper.__init__(self, **args)
        self.allowMpi = False
        self.allowThreads = False

    #--------------------------- DEFINE param functions -----------------------
    def _defineProcessParams(self, form):
        XmippOperateHelper._defineProcessParams(self, form)
    
    def _defineSpecificParams(self, form):
        form.addParam('inputParticles2', params.PointerParam,
                      allowNull=True,
                      condition=(binaryCondition+' and (not isValue) or '
                                 +dotCondition),
                      label='Input Particles (2nd)',
                      help = 'Set a SetOfParticles. The particles must be '
                             'the same dimensions as the input particles.',
                      pointerClass='SetOfParticles')
    
    #--------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        """ convert to Xmipp image model"""
        writeSetOfParticles(self.inputParticles.get(), self.inputFn,
                            alignType=ALIGN_NONE)
        
        if self.inputParticles2.get() is not None:
            writeSetOfParticles(self.inputParticles2.get(),
                                self._getSecondSetFn(),
                                alignType=ALIGN_NONE)
    
    def operationStep(self, operationStr):
        dictImgFn = {"inputFn" : self.inputFn}
        args = self._args  % dictImgFn + operationStr
        
        if self._isBinaryCond():
            if self.isValue:
                args += ' %f' % self.value.get()
            else:
                args += ' %s' % self._getSecondSetFn()
        elif self._isPowCond():
            args += ' %f' % self.value.get()
        elif self._isDotCond():
            args += ' %s' % self._getSecondSetFn()
        elif self._isintValueCond():
            args += ' %d' % self.intValue.get()
        
        args += " -o %s --save_metadata_stack %s" % (self.outputStk,
                                                     self.outputMd)
        args += " --keep_input_columns"
        self.runJob(self._program, args)
    
    #--------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        def _errorDimensions():
            if not (self.inputParticles.get() is None and
                    self.inputParticles2.get() is None):
                dim1 = self.inputParticles.get().getDimensions()
                dim2 = self.inputParticles2.get().getDimensions()
                if dim1 != dim2:
                    errors.append("The number of images in two operands are "
                                  "not the same. ")
        def _errorSize():
            if not (self.inputParticles.get() is None and
                    self.inputParticles2.get() is None):
                size1 = self.inputParticles.get().getSize()
                size2 = self.inputParticles2.get().getSize()
                if size1 != size2:
                    errors.append("Size of both *SetOfParticles* are not the"
                                  " same. ")
        
        if self._isBinaryCond():
            if not self.isValue:
                if not self.inputParticles2.get() is None:
                    _errorSize()
                    _errorDimensions()
        elif self._isDotCond():
            if not self.inputParticles2.get() is None:
                _errorDimensions()
        return errors


class XmippProtImageOperateVolumes(ProtOperateVolumes,
                                   XmippProcessVolumes,
                                   XmippOperateHelper):
    """ A Executes arithmetic operations between two volumesor to a volume. The
    mathematical operations are: plus, minus multiply, divide, minimum, maximum,
    dot product, log, log10, sqrt, abs, pow, column, slice, row, mn, radial
    average or resetThis enables structural comparisons, highlighting
    conformational changes or shared features between datasets.

    AI Generated

    ## Overview

    The Operate Volumes protocol applies mathematical operations to one volume, a
    set of volumes, or corresponding volumes from two inputs.

    The operation can be applied between a volume and a numerical value, between
    two volumes, between two sets of volumes, or as a unary operation on a single
    input.

    The protocol supports arithmetic operations such as addition, subtraction,
    multiplication, division, minimum, and maximum. It also supports dot product,
    logarithms, square root, absolute value, power, slice extraction, column and
    row extraction, radial average, and reset.

    The main output is a processed volume or volume set containing the result of
    the selected operation.

    ## Inputs and General Workflow

    The main input is one volume or a set of volumes.

    If the selected operation requires a second image operand, the user provides a
    second volume or volume set. If the operation uses a numerical value, the user
    enables the value option and provides the number.

    For volume sets, the protocol writes the input set to Xmipp metadata format.
    It then runs the Xmipp image-operation program and creates the corresponding
    output volume or output volume set.

    ## Input Volumes

    The **Input volumes** parameter defines the first operand of the operation.

    The input may be a single volume or a set of volumes. Unary operations act only
    on this input. Binary operations combine this input with either a numerical
    value or a second volume input.

    The protocol operates directly on voxel values.

    ## Operation

    The **Operation** parameter selects the mathematical operation.

    The available operations include:

    - plus;
    - minus;
    - multiply;
    - divide;
    - minimum;
    - maximum;
    - dot product;
    - log;
    - log10;
    - sqrt;
    - abs;
    - pow;
    - slice;
    - column;
    - row;
    - radial average;
    - reset.

    The selected operation determines whether a second volume, a numerical value,
    or an integer index is required.

    ## Binary Operations

    The binary operations are:

    - plus;
    - minus;
    - multiply;
    - divide;
    - minimum;
    - maximum.

    They can be applied either using a numerical value or using a second volume or
    volume set.

    For example, subtracting one volume from another produces a voxel-wise
    difference map. Multiplying a volume by a value scales all voxel intensities.

    ## Second Operand as a Value

    The **Second operand is a value?** option is used for binary operations.

    If enabled, the user provides the numerical **Input value**.

    For example:

    - plus 1 adds 1 to all voxels;
    - multiply 2 doubles the map intensity;
    - divide 10 scales the map down by a factor of 10.

    This is useful for simple map scaling or offset operations.

    ## Second Volume or Volume Set

    The **Input Volumes (2nd)** parameter is used when the operation requires a
    second image operand.

    If the first input is a single volume, the second input should be a compatible
    single volume.

    If the first input is a set of volumes, the second input should be a compatible
    set of volumes.

    For binary operations between volume sets, the two sets must have the same
    number of volumes and compatible dimensions. The protocol validates these
    conditions.

    ## Minimum and Maximum

    The **minimum** and **maximum** operations compute voxel-wise comparisons.

    With a numerical value, each voxel is compared with the value. With a second
    volume, each voxel is compared with the corresponding voxel in the second
    volume.

    These operations can be useful for clipping, masking-like effects, or combining
    maps by selecting lower or higher voxel values.

    ## Dot Product

    The **dot product** operation computes the dot product between two volumes or
    corresponding volumes.

    This operation requires a second volume input with compatible dimensions.

    It can be useful for technical workflows, similarity measurements, or
    quantitative comparisons.

    ## Logarithm Operations

    The **log** operation computes the natural logarithm of voxel values.

    The **log10** operation computes the decimal logarithm.

    These operations require suitable voxel values. Zero or negative values may
    produce invalid results.

    ## Square Root and Absolute Value

    The **sqrt** operation computes the square root of voxel values.

    The **abs** operation computes the absolute value.

    Square root should be used carefully with maps that contain negative density.
    Absolute value removes the sign of the density and therefore changes the
    meaning of positive and negative regions.

    ## Power Operation

    The **pow** operation raises voxel values to a selected exponent.

    The user provides the exponent through the **Input value** parameter.

    This operation can strongly modify map contrast and should be used cautiously,
    especially with negative values or non-integer exponents.

    ## Slice, Column, and Row

    The **slice**, **column**, and **row** operations extract a selected index from
    the volume data.

    The index is provided as an integer input value.

    These operations are mainly useful for technical inspection, debugging,
    conversion, or creating lower-dimensional views from a volume.

    ## Radial Average

    The **radial average** operation computes a radial average of the input image
    or volume.

    This can be useful for radial profiles, frequency-like summaries, or analysis
    of radially symmetric data.

    ## Reset

    The **reset** operation sets the image or volume values to zero.

    This is useful for technical workflows where a blank object with the same
    metadata structure is required.

    ## Output Volume or Volume Set

    The main output is the processed volume output produced by the Scipion volume
    processing framework.

    If the input is a single volume, the output is a single operated volume.

    If the input is a set of volumes, the output is a set of operated volumes whose
    items correspond to the input items.

    The output can be used in later protocols like any other Scipion volume object.

    ## Validation Rules

    When a binary operation uses a second volume set, both sets must have the same
    number of volumes and compatible dimensions.

    When dot product is selected, the two operands must have compatible dimensions.

    If these requirements are not met, the protocol reports validation errors.

    ## Interpreting the Result

    The output should be interpreted as a direct mathematical transformation of the
    input volume data.

    The protocol does not perform alignment, filtering, masking, refinement, or
    validation. It only applies the selected operation to voxel values.

    Operations such as subtraction can highlight differences between maps, but
    those differences should not automatically be interpreted as biological
    signal. They may also reflect misalignment, scaling differences, noise,
    filtering differences, or origin mismatch.

    ## Practical Recommendations

    Use value-based operations for simple scaling or offset correction.

    Use subtraction between two maps only when they are aligned, sampled
    consistently, and comparable in scale.

    Use minimum and maximum for controlled clipping or voxel-wise map combination.

    Use logarithm, square root, absolute value, and power operations only when the
    input value range makes sense.

    Use dot product only for compatible volumes intended for quantitative
    comparison.

    Inspect the output visually and, when comparing maps, check alignment and
    sampling before interpreting differences.

    ## Final Perspective

    Operate Volumes is a general volume-arithmetic utility.

    For biological users, its value is practical: it enables direct voxel-wise
    mathematical operations on maps or map sets. It can support preprocessing,
    map comparison, difference-map generation, normalization experiments, and
    technical workflows.

    The protocol should be treated as a low-level image-operation tool. It does
    not validate the biological meaning of the result; that interpretation depends
    on the input maps and on the operation selected by the user.
    """
    _label = 'operate volumes'
     
    def __init__(self, **args):
        ProtOperateVolumes.__init__(self, **args)
        XmippProcessVolumes.__init__(self)
        XmippOperateHelper.__init__(self, **args)
        self.allowMpi = False
        self.allowThreads = False

    
    #--------------------------- DEFINE param functions -----------------------
    def _defineProcessParams(self, form):
        XmippOperateHelper._defineProcessParams(self, form)
    
    def _defineSpecificParams(self, form):
        form.addParam('inputVolumes2', params.PointerParam,
                      allowNull=True,
                      condition=(binaryCondition + ' and (not isValue) or '
                                 + dotCondition),
                      label='Input Volumes (2nd)',
                      help = 'This parameter depends of the input volume(s). '
                             'If it is set a volume (or a SetOfVolumes) as '
                             'input, this must be a *Volume* (or '
                             '*SetOfVolumes*) object.',
                      pointerClass='Volume, SetOfVolumes')

    #--------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        """ convert to Xmipp image model"""
        if not self._isSingleInput():
            writeSetOfVolumes(self.inputVolumes.get(), self.inputFn)
            
            if self.inputVolumes2.get() is not None:
                writeSetOfVolumes(self.inputVolumes2.get(),
                                  self._getSecondSetFn())
    
    def operationStep(self, operationStr):
        dictImgFn = {"inputFn" : self.inputFn}
        args = self._args  % dictImgFn + operationStr
        
        if self._isBinaryCond():
            if self.isValue:
                args += ' %f' % self.value.get()
            else:
                args += ' %s' % self._getSecondVolumeFn()
        elif self._isPowCond():
            args += ' %f' % self.value.get()
        elif self._isDotCond():
            args += ' %s' % self._getSecondVolumeFn()
        elif self._isintValueCond():
            args += ' %d' % self.intValue.get()
        
        args += " -o %s " % self.outputStk
        if not self._isSingleInput():
            args += " --save_metadata_stack %s" \
                    " --keep_input_columns" % self.outputMd
        self.runJob(self._program, args)
    
    #--------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        def _errorDimensions():
            if not (self.inputVolumes.get() is None and
                    self.inputVolumes2.get() is None):
                dim1 = self.inputVolumes.get().getDimensions()
                dim2 = self.inputVolumes2.get().getDimensions()
                if dim1 != dim2:
                    errors.append("The number of volumes in two operands are "
                                  "not the same. ")
        def _errorSize():
            if (not (self.inputVolumes.get() is None and
                    self.inputVolumes2.get() is None) and
                    not self._isSingleInput()):
                size1 = self.inputVolumes.get().getSize()
                size2 = self.inputVolumes2.get().getSize()
                if size1 != size2:
                    errors.append("Size of both volumes are not the same. ")
        
        if self._isBinaryCond():
            if not self.isValue:
                if not self.inputVolumes2.get() is None:
                    _errorSize()
                    _errorDimensions()
        elif self._isDotCond():
            if not self.inputVolumes2.get() is None:
                _errorDimensions()
        return errors
    
    #--------------------------- UTILS functions ------------------------------
    def _getSecondVolumeFn(self):
        if self._isSingleInput():
            return getImageLocation(self.inputVolumes2.get())
        else:
            return self._getSecondSetFn()
