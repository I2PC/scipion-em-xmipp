# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco   (josue.gomez-blanco@mcgill.ca)
# *              Joaquin Oton   (oton@cnb.csic.es)
# *              Airen Zaldivar (azaldivar@cnb.csic.es)
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

import pyworkflow.protocol.constants as const
from pyworkflow.object import String
from pwem.convert.headers import setMRCSamplingRate
from pyworkflow.protocol.params import (BooleanParam, EnumParam, FloatParam,
                                        IntParam)
from pwem.objects import Volume, SetOfParticles, Mask

from .protocol_process import XmippProcessParticles, XmippProcessVolumes
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippResizeHelper:
    """ Common features to change dimensions of either SetOfParticles,
    Volume or SetOfVolumes objects.
    """
    _devStatus = PROD

    RESIZE_SAMPLINGRATE = 0
    RESIZE_DIMENSIONS = 1
    RESIZE_FACTOR = 2
    RESIZE_PYRAMID = 3

    WINDOW_OP_CROP = 0
    WINDOW_OP_WINDOW = 1
    
    #--------------------------- DEFINE param functions --------------------------------------------       
    @classmethod   
    def _defineProcessParams(cls, protocol, form):
        # Resize operation
        form.addParam('doResize', BooleanParam, default=False,
                      label='Resize %s?' % protocol._inputLabel,
                      help='If you set to *Yes*, you should provide a resize option.')
        form.addParam('resizeOption', EnumParam,
                      choices=['Sampling Rate', 'Dimensions', 'Factor', 'Pyramid'],
                      condition='doResize',
                      default=cls.RESIZE_SAMPLINGRATE,
                      label="Resize option", display=EnumParam.DISPLAY_COMBO,
                      help='Select an option to resize the images: \n '
                      '_Sampling Rate_: Set the desire sampling rate to resize. \n'
                      '_Dimensions_: Set the output dimensions. Resize operation can be done in Fourier space.\n'
                      '_Factor_: Set a resize factor to resize. \n '
                      '_Pyramid_: Use positive level value to expand and negative to reduce. \n'
                      'Pyramid uses spline pyramids for the interpolation. All the rest uses normally interpolation\n'
                      '(cubic B-spline or bilinear interpolation). If you set the method to dimensions, you may choose\n'
                      'between interpolation and Fourier cropping.')
        form.addParam('resizeSamplingRate', FloatParam, default=1.0,
                      condition='doResize and resizeOption==%d' % cls.RESIZE_SAMPLINGRATE,
                      allowsPointers=True,
                      label='Resize sampling rate (Å/px)',
                      help='Set the new output sampling rate.')
        form.addParam('doFourier', BooleanParam, default=False,
                      condition='doResize and resizeOption==%d' % cls.RESIZE_DIMENSIONS,
                      label='Use fourier method to resize?',
                      help='If you set to *True*, the final dimensions must be lower than the original ones.')
        form.addParam('resizeDim', IntParam, default=0,
                      condition='doResize and resizeOption==%d' % cls.RESIZE_DIMENSIONS,
                      allowsPointers=True,
                      label='New image size (px)',
                      help='Size in pixels of the particle images <x> <y=x> <z=x>.')
        form.addParam('resizeFactor', FloatParam, default=0.5,
                      condition='doResize and resizeOption==%d' % cls.RESIZE_FACTOR,
                      allowsPointers=True,
                      label='Resize factor',
                      help='New size is the old one x resize factor.')
        form.addParam('resizeLevel', IntParam, default=0,
                      condition='doResize and resizeOption==%d' % cls.RESIZE_PYRAMID,
                      allowsPointers=True,
                      label='Pyramid level',
                      help='Use positive value to expand and negative to reduce.')
        form.addParam('hugeFile', BooleanParam, default=False, expertLevel=const.LEVEL_ADVANCED,
                      label='Huge file',
                      help='If the file is huge, very likely you may have problems doing the antialiasing filter '
                           '(because there is no memory for the input and its Fourier tranform). This option '
                           'removes the antialiasing filter (meaning you will get aliased results), and performs '
                           'a bilinear interpolation (to avoid having to produce the B-spline coefficients).')
        # Window operation
        form.addParam('doWindow', BooleanParam, default=False,
                      label='Apply a window operation?',
                      help='If you set to *Yes*, you should provide a window option.')
        form.addParam('windowOperation', EnumParam,
                      choices=['crop', 'window'],
                      condition='doWindow',
                      default=cls.WINDOW_OP_WINDOW,
                      label="Window operation", display=EnumParam.DISPLAY_COMBO,
                      help='Select how to change the size of the particles.\n'
                      '_cls.RESIZE_: provide the new size (in pixels) for your particles.\n'
                      '_crop_: choose how many pixels to crop from each border.\n')
        form.addParam('cropSize', IntParam, default=0,
                      condition='doWindow and windowOperation == %d' % cls.WINDOW_OP_CROP,
                      allowsPointers=True,
                      label='Crop size (px)',
                      help='Amount of pixels cropped from each border.\n'
                           'e.g: if you set 10 pixels, the dimensions of the\n'
                           'object (SetOfParticles, Volume or SetOfVolumes) will be\n'
                           'reduced in 20 pixels (2 borders * 10 pixels)')
        form.addParam('windowSize', IntParam, default=0,
                      allowsPointers=True,
                      condition='doWindow and windowOperation == %d' % cls.WINDOW_OP_WINDOW,
                      label='Window size (px)',
                      help='Size in pixels of the output object. It will be '
                           'expanded or cutted in all directions such that the '
                           'origin remains the same.')

    #--------------------------- INSERT steps functions ------------------------
    @classmethod
    def _insertProcessStep(cls, protocol):
        isFirstStep = True
        
        if protocol.doResize:
            args = protocol._resizeArgs()
            if protocol.samplingRate>protocol.samplingRateOld and not protocol.hugeFile:
                protocol._insertFunctionStep("filterStep", isFirstStep, protocol._filterArgs())
                isFirstStep = False
            protocol._insertFunctionStep("resizeStep", isFirstStep, args)
            isFirstStep = False
            
        if protocol.doWindow:
            protocol._insertFunctionStep("windowStep", isFirstStep, protocol._windowArgs())
    
    #--------------------------- INFO functions --------------------------------
    @classmethod
    def _validate(cls, protocol):
        errors = []
        
        if (protocol.doResize and protocol.doFourier and
            protocol.resizeOption == cls.RESIZE_SAMPLINGRATE):
            size = protocol._getSetSize()
            if protocol.resizeDim > size:
                errors.append('Fourier resize method cannot be used to '
                              'increase the dimensions')
                
        return errors
    
    #--------------------------- STEP functions --------------------------------
    @classmethod
    def filterStep(cls, protocol, args):
        protocol.runJob("xmipp_transform_filter", args)
    
    @classmethod
    def resizeStep(cls, protocol, args):
        protocol.runJob("xmipp_image_resize", args)
    
    @classmethod
    def windowStep(cls, protocol, args):
        protocol.runJob("xmipp_transform_window", args, numberOfMpi=1)
        

    #--------------------------- UTILS functions -------------------------------
    @classmethod
    def _filterCommonArgs(cls, protocol):
        return "--fourier low_pass %f"%\
            (protocol.samplingRateOld/(2*protocol.samplingRate))

    @classmethod
    def _resizeCommonArgs(cls, protocol):
        samplingRate = protocol._getSetSampling()
        
        if protocol.resizeOption == cls.RESIZE_SAMPLINGRATE:
            newSamplingRate = protocol.resizeSamplingRate.get()
            factor = samplingRate / newSamplingRate
            args = " --factor %(factor)f"
        elif protocol.resizeOption == cls.RESIZE_DIMENSIONS:
            size = protocol.resizeDim.get()
            dim = protocol._getSetSize()
            factor = float(size) / float(dim)
            newSamplingRate = samplingRate / factor
            
            if protocol.doFourier and not protocol.hugeFile:
                args = " --fourier %(size)d"
            else:
                args = " --dim %(size)d"
        elif protocol.resizeOption == cls.RESIZE_FACTOR:
            factor = protocol.resizeFactor.get()
            newSamplingRate = samplingRate / factor
            args = " --factor %(factor)f"
        elif protocol.resizeOption == cls.RESIZE_PYRAMID:
            level = protocol.resizeLevel.get()
            factor = 2**level
            newSamplingRate = samplingRate / factor
            args = " --pyramid %(level)d"
        if protocol.hugeFile:
            args+=" --interp linear"
            
        protocol.samplingRate = newSamplingRate
        protocol.samplingRateOld = samplingRate
        protocol.factor = factor
        
        return args % locals()
    
    @classmethod
    def _windowCommonArgs(cls, protocol):
        op = protocol.getEnumText('windowOperation')
        if op == "crop":
            cropSize2 = protocol.cropSize.get() * 2
            protocol.newWindowSize = protocol._getSetSize() - cropSize2
            return " --crop %d " % cropSize2
        elif op == "window":
            windowSize = protocol.windowSize.get()
            protocol.newWindowSize = windowSize
            return " --size %d " % windowSize


def _getSize(imgSet):
    """ get the size of an object"""
    if isinstance(imgSet, Volume):
        Xdim = imgSet.getDim()[0]
    else:
        Xdim = imgSet.getDimensions()[0]
    return Xdim

def _getSampling(imgSet):
    """ get the sampling rate of an object"""
    samplingRate = imgSet.getSamplingRate()
    return samplingRate


class XmippProtCropResizeParticles(XmippProcessParticles):
    """ Crop or resize a set of particles.

    AI Generated

    ## Overview

    The Crop/Resize Particles protocol changes the image size or sampling rate of
    a particle set.

    This protocol is useful when particles need to be prepared for another
    processing step that expects a different box size or pixel size. It can resize
    particles, crop or window them, or perform both operations in sequence.

    The protocol can also process a 2D mask instead of a particle set. In that
    case, the output is a resized or cropped mask rather than particles.

    When particles are resized, the protocol updates the sampling rate. If the
    particles contain coordinates or alignment transformations, these are also
    scaled so that they remain consistent with the resized images.

    ## Inputs and General Workflow

    The input can be:

    - a set of particles;
    - a 2D mask.

    The protocol can perform two types of operation:

    - resize;
    - window operation.

    If both operations are selected, resizing is performed first and the window
    operation is applied afterwards.

    For particles, the protocol converts the input set to Xmipp metadata, applies
    the selected operations, and creates a new output particle set. For masks, it
    applies the selected operations to the mask image and creates a new output
    mask.

    ## Input Particles or Mask

    The **Input particles/Mask** parameter defines the object to be processed.

    If the input is a particle set, the output is **outputParticles**.

    If the input is a 2D mask, the output is **outputMask**.

    The protocol keeps the input information whenever possible, but updates image
    locations and sampling-related metadata according to the selected operations.

    ## Resize Particles

    The **Resize particles?** option enables image resizing.

    When this option is enabled, the user must choose a resize method. Resizing
    changes the particle dimensions and the sampling rate consistently.

    For example, if particles are downsampled by a factor of 0.5, the box size
    becomes smaller and the sampling rate becomes larger.

    Resizing is useful to reduce computational cost, match another dataset, prepare
    particles for neural-network methods, or create lower-resolution working
    copies.

    ## Resize Option

    The **Resize option** parameter defines how the new size is specified.

    The available options are:

    **Sampling Rate**: the user provides the desired output sampling rate.

    **Dimensions**: the user provides the desired output image size in pixels.

    **Factor**: the user provides a multiplicative resize factor.

    **Pyramid**: the user provides a pyramid level. Positive values expand the
    image and negative values reduce it.

    The chosen option determines how the output size and output sampling rate are
    computed.

    ## Resize by Sampling Rate

    When **Sampling Rate** is selected, the user provides the desired output pixel
    size in angstroms per pixel.

    The protocol computes the corresponding resize factor from the current sampling
    rate and the requested sampling rate.

    This option is useful when the user wants the output particles to match a
    specific physical pixel size.

    ## Resize by Dimensions

    When **Dimensions** is selected, the user provides the desired output image
    size in pixels.

    The protocol computes the corresponding resize factor from the input size and
    the requested size. It also updates the sampling rate accordingly.

    This option is useful when a downstream protocol requires a specific box size,
    for example 128, 256, or 512 pixels.

    ## Fourier Resize

    The **Use Fourier method to resize?** option is available when resizing by
    dimensions.

    If enabled, resizing is performed in Fourier space. This is appropriate for
    downsampling, but it cannot be used to increase the image dimensions.

    If disabled, the protocol uses interpolation-based resizing.

    Fourier resizing is often preferred for reducing image size because it can
    preserve frequency-domain behavior more naturally. However, it should only be
    used when the final dimensions are smaller than the original ones.

    ## Resize by Factor

    When **Factor** is selected, the user provides a multiplicative resize factor.

    For example:

    - a factor of 0.5 halves the image size;
    - a factor of 2 doubles the image size.

    The sampling rate is updated inversely to the factor. Reducing the image size
    increases the angstrom-per-pixel value, while enlarging the image decreases it.

    ## Resize by Pyramid

    When **Pyramid** is selected, the protocol uses spline-pyramid interpolation.

    Positive pyramid levels expand the image, while negative levels reduce it.

    This option is useful when the user wants pyramid-based interpolation rather
    than the standard resize methods.

    ## Huge File Option

    The **Huge file** option is intended for very large files.

    When enabled, the protocol avoids the antialiasing filter and uses linear
    interpolation. This can reduce memory requirements, but it may introduce
    aliasing artifacts.

    This option should be used only when memory limitations prevent the standard
    processing from running.

    ## Antialiasing Filter

    When downsampling to a larger sampling rate, the protocol can apply a low-pass
    filter before resizing, unless **Huge file** is enabled.

    This antialiasing step reduces high-frequency content that would otherwise be
    folded into lower frequencies during downsampling.

    It is usually desirable when reducing image size.

    ## Apply a Window Operation

    The **Apply a window operation?** option changes the particle or mask box size
    by cropping or windowing.

    This operation can be used alone or after resizing.

    Windowing is useful when the user wants a specific final box size, while
    cropping is useful when the user wants to remove a fixed number of pixels from
    the borders.

    ## Crop Operation

    When the window operation is **crop**, the **Crop size** parameter defines how
    many pixels are removed from each border.

    For example, if the crop size is 10 pixels, the final image size is reduced by
    20 pixels in each dimension because 10 pixels are removed from both sides.

    Cropping is useful for removing empty borders or reducing the box around a
    centered particle.

    ## Window Operation

    When the window operation is **window**, the **Window size** parameter defines
    the final output size in pixels.

    The image is expanded or cut in all directions so that the origin remains
    consistent.

    This is useful when the user needs a precise final box size for downstream
    processing.

    ## Metadata Updates for Particles

    When particles are resized, the protocol updates important metadata.

    The output sampling rate is changed to the new value. If particles have
    coordinates, the coordinates are scaled by the resize factor. If particles have
    alignment transformations, the shifts in the transformation matrices are also
    scaled.

    These updates are important because they keep the particle metadata consistent
    with the resized images.

    ## Output Particles

    If the input is a particle set, the main output is **outputParticles**.

    This output contains the processed particles after resizing, cropping, or
    windowing. The output particle set preserves the input metadata and updates
    image locations, sampling rate, coordinates, and alignment shifts when needed.

    The output can be used in downstream protocols like any other particle set.

    ## Output Mask

    If the input is a 2D mask, the main output is **outputMask**.

    This output contains the resized or cropped mask. It copies the input mask
    information and updates the sampling rate when resizing is performed.

    This is useful when the same mask needs to be adapted to match resized or
    windowed particle images.

    ## Validation Rules

    The protocol requires that at least one operation is selected: resizing,
    windowing, or both.

    If no operation is selected, validation reports an error.

    The protocol also prevents invalid Fourier resizing cases where the Fourier
    method would be used to increase the image dimensions.

    ## Practical Recommendations

    Use resizing to reduce computational cost or to match a target pixel size.

    Use Fourier resizing when downsampling by dimensions and memory allows it.

    Use the huge-file option only when memory limitations require it.

    Use cropping when you want to remove a fixed number of pixels from the borders.

    Use windowing when you need a precise final box size.

    When resizing particles with coordinates or alignments, rely on this protocol
    rather than manually resizing images, because it updates the associated
    metadata.

    Inspect a subset of output particles or masks to confirm that the final size
    and centering are correct.

    ## Final Perspective

    Crop/Resize Particles is a preparation protocol for changing particle or 2D
    mask dimensions.

    For biological users, its value is that it adapts particle images to the size
    and sampling requirements of downstream workflows while preserving consistent
    metadata. This is especially important when particles have coordinates or
    alignment shifts that must remain compatible with the processed images.
    """
    # Protocol constants
    OUTPUT_PARTICLES_NAME = 'outputParticles'
    OUTPUT_MASK_NAME = 'outputMask'
    _devStatus = PROD

    _label = 'crop/resize particles'
    _inputLabel = 'particles'
    _possibleOutputs = {OUTPUT_PARTICLES_NAME: SetOfParticles, OUTPUT_MASK_NAME: Mask}
    
    def __init__(self, **kwargs):
        XmippProcessParticles.__init__(self, **kwargs)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        # Creating super form
        super()._defineParams(form)
       
        # Obtaining input particles param to accept also a mask
        inputParticles = form.getParam('inputParticles')
        inputParticles.pointerClass = String(str(inputParticles.pointerClass) + ',Mask')
        inputParticles.label = String(str(inputParticles.label) + '/Mask')
        inputParticles.help = String('Input particles or 2D Mask to be cropped/resized.')

    def _defineProcessParams(self, form):
        XmippResizeHelper._defineProcessParams(self, form)
        form.addParallelSection(threads=0, mpi=8)
        
    def _insertProcessStep(self):
        XmippResizeHelper._insertProcessStep(self)
     
    #--------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, isFirstStep, args):
        XmippResizeHelper.filterStep(self, self._ioArgs(isFirstStep)+args)
    
    def resizeStep(self, isFirstStep, args):
        XmippResizeHelper.resizeStep(self, self._ioArgs(isFirstStep)+args)
    
    def windowStep(self, isFirstStep, args):
        XmippResizeHelper.windowStep(self, self._ioArgs(isFirstStep)+args)
    
    def convertInputStep(self):
        """ convert if necessary"""
        if self.isMask():
            # If input is a Mask, modify filter params
            self.inputFn = self.inputParticles.get().getFileName()
            inputName = os.path.splitext(os.path.basename(self.inputFn))[0]

            # Set output mask path
            self.outputStk = self._getExtraPath(os.path.basename(inputName + '.mrc'))
            self.outputMd = self._getTmpPath('tmp.xmd')
        else:
            # If input is not Mask, keep default behaviour
            super().convertInputStep()
    
    def createOutputStep(self):
        if self.isMask():
            # If input is a Mask, create output Mask
            outputMask = Mask(self.outputStk)
            outputMask.copyInfo(self.inputParticles.get())
            self._preprocessOutput(outputMask)
            self._defineOutputs(**{self.OUTPUT_MASK_NAME: outputMask})
            self._defineTransformRelation(self.inputParticles.get(), outputMask)
        else:
            # If input is not Mask, keep default behaviour
            super().createOutputStep()
        
    def _preprocessOutput(self, output):
        """
        We need to update the sampling rate of the 
        particles if the Resize option was used.
        """
        if not self.isMask():
            self.inputHasAlign = self.inputParticles.get().hasAlignment()
        
        if self.doResize:
            output.setSamplingRate(self.samplingRate)
            setMRCSamplingRate(self.outputStk, self.samplingRate)
            
    def _updateItem(self, item, row):
        """ Update also the sampling rate and 
        the alignment if needed.
        """
        XmippProcessParticles._updateItem(self, item, row)
        if self.doResize:
            if item.hasCoordinate():
                item.scaleCoordinate(self.factor)
            item.setSamplingRate(self.samplingRate)
            if self.inputHasAlign:
                item.getTransform().scaleShifts(self.factor)
    
    #--------------------------- INFO functions ----------------------------------------------------
    def _summary(self):
        summary = []
        
        if not hasattr(self, 'outputParticles'):
            summary.append("Output images not ready yet.") 
        else:
            sampling = _getSampling(self.outputParticles)
            size = _getSize(self.outputParticles)
            if self.doResize:
                summary.append(u"Output particles have a different sampling "
                               u"rate (pixel size): *%0.3f* Å/px" % sampling)
                summary.append("Resizing method: *%s*" %
                               self.getEnumText('resizeOption'))
            if self.doWindow:
                if self.getEnumText('windowOperation') == "crop":
                    summary.append("The particles were cropped.")
                else:
                    summary.append("The particles were windowed.")
                summary.append("New size: *%s* px" % size)
        return summary

    def _methods(self):

        if not hasattr(self, 'outputParticles'):
            return []

        methods = ["We took input particles %s of size %d " % (self.getObjectTag('inputParticles'), len(self.inputParticles.get()))]
        if self.doWindow:
            if self.getEnumText('windowOperation') == "crop":
                methods += ["cropped them"]
            else:
                methods += ["windowed them"]
        if self.doResize:
            outputParticles = getattr(self, 'outputParticles', None)
            if outputParticles is None or outputParticles.getDim() is None:
                methods += ["Output particles not ready yet."]
            else:
                methods += ['resized them to %d px using the "%s" method%s' %
                            (outputParticles.getDim()[0],
                             self.getEnumText('resizeOption'),
                             " in Fourier space" if self.doFourier else "")]
        if not self.doResize and not self.doWindow:
            methods += ["did nothing to them"]
        str = "%s and %s. Output particles: %s" % (", ".join(methods[:-1]),
                                                   methods[-1],
                                                   self.getObjectTag('outputParticles'))
        return [str]

    def _validate(self):
        """ This function validates the input parameters so only allowed operations take place. """
        # Getting default errors
        errors = XmippResizeHelper._validate(self)

        # Checking if at least one of the operations has been selected
        if not self.doResize and not self.doWindow.get():
            errors.append('At least one of the possible operations needs to be selected.')
        
        # Returning errors
        return errors
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def isMask(self):
        """ This function returns True if the input object is a Mask. False otherwise. """
        return isinstance(self.inputParticles.get(), Mask)
      
    def _ioArgs(self, isFirstStep):
        if isFirstStep:
            return "-i %s -o %s --save_metadata_stack %s --keep_input_columns " % (self.inputFn, self.outputStk, self.outputMd)
        else:
            return "-i %s " % self.outputStk

    def _filterArgs(self):
        return XmippResizeHelper._filterCommonArgs(self)

    def _resizeArgs(self):
        return XmippResizeHelper._resizeCommonArgs(self)
    
    def _windowArgs(self):
        return XmippResizeHelper._windowCommonArgs(self)
    
    def _getSetSize(self):
        """ get the size of SetOfParticles object"""
        imgSet = self.inputParticles.get()
        return _getSize(imgSet)
    
    def _getSetSampling(self):
        """ get the sampling rate of SetOfParticles object"""
        imgSet = self.inputParticles.get()
        return _getSampling(imgSet)
    
    def _getDefaultParallel(self):
        """ Return the default value for thread and MPI
        for the parallel section definition.
        """
        return (0, 1)



class XmippProtCropResizeVolumes(XmippProcessVolumes):
    """Crops or resizes 3D volumes to a desired size or region of interest.
    This protocol helps optimize memory usage and focus on relevant structural
    areas for analysis or comparison.

    AI Generated

    ## Overview

    The Crop/Resize Volumes protocol changes the size or sampling rate of one
    volume or a set of volumes.

    This protocol is useful when maps need to be downsampled, resampled to a target
    pixel size, cropped to remove borders, windowed to a new box size, or prepared
    for comparison with other volumes. It can perform resizing, window operations,
    or both.

    When a single input volume has associated half maps, the protocol applies the
    same filtering, resizing, and windowing operations to the half maps as well.
    This keeps the full map and its half maps geometrically consistent.

    ## Inputs and General Workflow

    The input can be:

    - a single volume;
    - a set of volumes.

    The protocol can perform two types of operation:

    - resize;
    - window operation.

    If both are enabled, resizing is performed first and the window operation is
    applied afterwards.

    The protocol writes the processed maps and creates the corresponding Scipion
    output volume or volume set. Sampling rate and origin information are updated
    when resizing is performed.

    ## Input Volumes

    The **Input volumes** parameter defines the map or maps to be processed.

    If the input is a single volume, the output is a processed single volume.

    If the input is a set of volumes, the output is a processed volume set.

    The protocol is designed for geometrical preparation of maps. It does not
    perform alignment, refinement, masking, sharpening, or validation.

    ## Resize Volumes

    The **Resize volumes?** option enables resizing.

    When enabled, the user chooses how to define the new size or sampling rate. The
    output sampling rate is updated consistently with the selected resize factor.

    Resizing is useful when the user wants to reduce memory usage, match maps from
    different workflows, create lower-resolution working copies, or prepare maps
    for algorithms requiring a specific voxel size.

    ## Resize Option

    The **Resize option** parameter defines how the resizing is specified.

    The available options are:

    **Sampling Rate**: define the desired output sampling rate.

    **Dimensions**: define the desired output box size in pixels.

    **Factor**: define a multiplicative resize factor.

    **Pyramid**: define a spline-pyramid level.

    Each option leads to a consistent update of both map dimensions and sampling
    rate.

    ## Resize by Sampling Rate

    When **Sampling Rate** is selected, the user provides the new voxel size in
    angstroms per pixel.

    The protocol computes the resize factor from the old sampling rate and the new
    one.

    This option is useful when volumes need to match a specific physical sampling
    for downstream comparison or processing.

    ## Resize by Dimensions

    When **Dimensions** is selected, the user provides the final box size in
    pixels.

    The protocol computes the resize factor and updates the sampling rate
    accordingly.

    This option is useful when a downstream protocol requires a specific cubic box
    size.

    ## Fourier Resize

    The **Use Fourier method to resize?** option is available when resizing by
    dimensions.

    If enabled, resizing is performed in Fourier space. This is intended for
    reducing the volume size and should not be used to increase dimensions.

    If disabled, interpolation-based resizing is used.

    Fourier resizing can be appropriate when downsampling maps while preserving
    frequency-domain properties.

    ## Resize by Factor

    When **Factor** is selected, the user provides a resize factor.

    A factor below 1 reduces the map dimensions. A factor above 1 enlarges them.

    The sampling rate is updated inversely to the factor.

    ## Resize by Pyramid

    When **Pyramid** is selected, the protocol uses spline pyramids.

    Positive levels expand the map and negative levels reduce it.

    This option is useful for pyramid-based interpolation workflows.

    ## Huge File Option

    The **Huge file** option is intended for very large volume files.

    When enabled, the protocol disables the antialiasing filter and uses linear
    interpolation. This reduces memory usage but can introduce aliasing artifacts.

    Use this option only when the standard method cannot be run because of memory
    limitations.

    ## Antialiasing Filter

    When downsampling to a larger sampling rate, the protocol can apply a
    low-pass filter before resizing, unless the huge-file option is enabled.

    This prevents high-frequency information from aliasing into lower frequencies.

    The same filter is applied to associated half maps when processing a single
    volume with half maps.

    ## Apply a Window Operation

    The **Apply a window operation?** option changes the box size of the volume or
    volume set.

    The operation can be:

    - crop;
    - window.

    Window operations can be performed alone or after resizing.

    ## Crop Operation

    When the window operation is **crop**, the **Crop size** parameter defines how
    many pixels are removed from each border.

    If the crop size is 10 pixels, the final box size is reduced by 20 pixels in
    each dimension.

    Cropping is useful for removing empty borders or reducing box size around a
    centered map.

    ## Window Operation

    When the window operation is **window**, the **Window size** parameter defines
    the final output box size in pixels.

    The protocol expands or cuts the volume in all directions while preserving the
    origin convention as consistently as possible.

    This option is useful when the map must have a specific final box size.

    ## Half-Map Processing

    If the input is a single volume and it has associated half maps, the protocol
    applies the same operations to the half maps.

    This includes filtering, resizing, and windowing.

    This behavior is important because many validation and post-processing
    workflows require the full map and half maps to have the same dimensions,
    sampling rate, and preprocessing history.

    ## Output Sampling Rate

    When resizing is performed, the output sampling rate is updated to the new
    computed value.

    For a set of volumes, the output set receives the new sampling rate.

    For a single volume, the output volume receives the new sampling rate, and the
    MRC header is also updated.

    ## Origin Update for Single Volumes

    When resizing a single volume, the protocol adjusts the origin shifts so that
    the physical coordinate convention remains consistent with the new sampling
    rate and output dimensions.

    This is important because changing the number of voxels and voxel size can
    otherwise shift the apparent position of the map in physical space.

    The output volume origin is therefore updated from the input origin, input
    dimensions, output dimensions, input sampling, and output sampling.

    ## Output Volume or Volume Set

    The main output is **outputVol**.

    If the input is a single volume, this output is the processed volume.

    If the input is a set of volumes, this output is the processed volume set.

    The output can be used in downstream protocols like any other Scipion volume
    or volume set.

    ## Validation Rules

    The protocol prevents invalid Fourier resizing cases where Fourier resizing
    would be used to increase the dimensions.

    For volumes, the shared validation is applied to resizing choices.

    Users should additionally ensure that window sizes and crop sizes are
    reasonable for the input volume dimensions.

    ## Practical Recommendations

    Use resizing by sampling rate when the goal is to match a specific voxel size.

    Use resizing by dimensions when the goal is to obtain a specific box size.

    Use Fourier resizing for downsampling when appropriate.

    Use cropping to remove a fixed border region.

    Use windowing to enforce a precise final box size.

    Use the huge-file option only when memory limitations prevent normal
    processing.

    For volumes with half maps, this protocol is useful because it keeps the full
    map and half maps processed consistently.

    After resizing, check the output sampling rate and origin before using the map
    for alignment, validation, or map-model comparison.

    ## Final Perspective

    Crop/Resize Volumes is a map-preparation protocol.

    For biological users, its value is that it adapts maps to the size and voxel
    spacing required by downstream processing while preserving important metadata
    such as sampling rate, origin, and, when present, half-map consistency.

    The protocol should be understood as a geometrical and sampling-rate
    transformation step. It does not change the biological content of the map, but
    it prepares the map for more efficient or compatible processing.
    """
    _label = 'crop/resize volumes'
    _inputLabel = 'volumes'
    _devStatus = PROD

    def __init__(self, **kwargs):
        XmippProcessVolumes.__init__(self, **kwargs)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippResizeHelper._defineProcessParams(self, form)
        
    def _insertProcessStep(self):
        XmippResizeHelper._insertProcessStep(self)

    #--------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, isFirstStep, args):
        XmippResizeHelper.filterStep(self, self._ioArgs(isFirstStep)+args)
        if self._isSingleInput() and self.inputVolumes.get().hasHalfMaps():
            XmippResizeHelper.filterStep(self, self._ioArgsHalf(isFirstStep,0) + args)
            XmippResizeHelper.filterStep(self, self._ioArgsHalf(isFirstStep,1) + args)

    def resizeStep(self, isFirstStep, args):
        XmippResizeHelper.resizeStep(self, self._ioArgs(isFirstStep)+args)
        if self._isSingleInput() and self.inputVolumes.get().hasHalfMaps():
            XmippResizeHelper.resizeStep(self, self._ioArgsHalf(isFirstStep,0) + args)
            XmippResizeHelper.resizeStep(self, self._ioArgsHalf(isFirstStep,1) + args)

    def windowStep(self, isFirstStep, args):
        XmippResizeHelper.windowStep(self, self._ioArgs(isFirstStep)+args)
        if self._isSingleInput() and self.inputVolumes.get().hasHalfMaps():
            XmippResizeHelper.windowStep(self, self._ioArgsHalf(isFirstStep,0) + args)
            XmippResizeHelper.windowStep(self, self._ioArgsHalf(isFirstStep,1) + args)

    def _preprocessOutput(self, volumes):
        # We use the preprocess only when input is a set
        # we do not use postprocess to setup correctly
        # the samplingRate before each volume is added
        if not self._isSingleInput():
            if self.doResize:
                volumes.setSamplingRate(self.samplingRate)

    def _postprocessOutput(self, volume:Volume):
        # We use the postprocess only when input is a volume
        if self._isSingleInput():
            if self.doResize:
                volume.setSamplingRate(self.samplingRate)
                # we have a new sampling so origin need to be adjusted
                iSampling = self.inputVolumes.get().getSamplingRate()
                oSampling = self.samplingRate
                xdim_i, ydim_i, zdim_i = self.inputVolumes.get().getDim()
                xdim_o, ydim_o, zdim_o = volume.getDim()

                xOrig, yOrig , zOrig = \
                    self.inputVolumes.get().getShiftsFromOrigin()
                xOrig += (xdim_i*iSampling-xdim_o*oSampling)/2.
                yOrig += (ydim_i*iSampling-ydim_o*oSampling)/2.
                zOrig += (zdim_i*iSampling-zdim_o*oSampling)/2.
                volume.setShiftsInOrigin(xOrig, yOrig, zOrig)
                volume.setSamplingRate(oSampling)
                setMRCSamplingRate(volume.getFileName(), oSampling)

    #--------------------------- INFO functions ----------------------------------------------------
    def _summary(self):
        summary = []
        
        if not hasattr(self, 'outputVol'):
            summary.append("Output volume(s) not ready yet.") 
        else:
            sampling = _getSampling(self.outputVol)
            size = _getSize(self.outputVol)
            if self.doResize:
                summary.append(u"Output volume(s) have a different sampling "
                               u"rate (pixel size): *%0.3f* Å/px" % sampling)
                summary.append("Resizing method: *%s*" %
                               self.getEnumText('resizeOption'))
            if self.doWindow.get():
                if self.getEnumText('windowOperation') == "crop":
                    summary.append("The volume(s) were cropped.")
                else:
                    summary.append("The volume(s) were windowed.")
                summary.append("New size: *%s* px" % size)
        return summary

    def _methods(self):
        if not hasattr(self, 'outputVol'):
            return []

        if self._isSingleInput():
            methods = ["We took one volume"]
            pronoun = "it"
        else:
            methods = ["We took %d volumes" % self.inputVolumes.get().getSize()]
            pronoun = "them"
        if self.doWindow.get():
            if self.getEnumText('windowOperation') == "crop":
                methods += ["cropped %s" % pronoun]
            else:
                methods += ["windowed %s" % pronoun]
        if self.doResize:
            outputVol = getattr(self, 'outputVol', None)
            if outputVol is None or self.outputVol.getDim() is None:
                methods += ["Output volume not ready yet."]
            else:
                methods += ['resized %s to %d px using the "%s" method%s' %
                            (pronoun, self.outputVol.getDim()[0],
                             self.getEnumText('resizeOption'),
                             " in Fourier space" if self.doFourier else "")]
        if not self.doResize and not self.doWindow:
            methods += ["did nothing to %s" % pronoun]
            # TODO: does this case even work in the protocol?
        return ["%s and %s." % (", ".join(methods[:-1]), methods[-1])]

    def _validate(self):
        return XmippResizeHelper._validate(self)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _ioArgs(self, isFirstStep):
        if isFirstStep:
            if self._isSingleInput():
                return "-i %s -o %s " % (self.inputFn, self.outputStk)
            else:
                return "-i %s -o %s --save_metadata_stack %s --keep_input_columns " % (self.inputFn, self.outputStk, self.outputMd)
        else:
            return "-i %s" % self.outputStk

    def _ioArgsHalf(self, isFirstStep, halfIdx=0):
        localHalves = [self._getExtraPath("half1.mrc"), self._getExtraPath("half2.mrc")]
        if isFirstStep:
            inputVol = self.inputVolumes.get()
            fnHalves = inputVol.getHalfMaps().split(',')
            return "-i %s -o %s " % (fnHalves[halfIdx], localHalves[halfIdx])
        else:
            return "-i %s"%localHalves[halfIdx]

    def _filterArgs(self):
        return XmippResizeHelper._filterCommonArgs(self)

    def _resizeArgs(self):
        return XmippResizeHelper._resizeCommonArgs(self)
    
    def _windowArgs(self):
        return XmippResizeHelper._windowCommonArgs(self)
    
    def _getSetSize(self):
        """ get the size of either Volume or SetOfVolumes objects"""
        imgSet = self.inputVolumes.get()
        size = _getSize(imgSet)
        return size
    
    def _getSetSampling(self):
        """ get the sampling rate of either Volume or SetOfVolumes objects"""
        imgSet = self.inputVolumes.get()
        samplingRate = _getSampling(imgSet)
        return samplingRate
