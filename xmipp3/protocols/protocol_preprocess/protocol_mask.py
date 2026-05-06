# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Josue Gomez Blanco     (josue.gomez-blanco@mcgill.ca)
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

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtMaskParticles, ProtMaskVolumes
from pyworkflow.protocol.params import EnumParam, PointerParam, IntParam

from xmipp3.convert import getImageLocation
from xmipp3.constants import *
from .geometrical_mask import XmippGeometricalMask3D, XmippGeometricalMask2D
from .protocol_process import XmippProcessParticles, XmippProcessVolumes
from .protocol_create_mask3d import XmippProtCreateMask3D


class XmippProtMask:
    """ This class implement the common features for applying a mask with
    Xmipp either SetOfParticles, Volume or SetOfVolumes objects.
    """
    
    def __init__(self, **args):
        self._program = "xmipp_transform_mask"
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        """ Add common mask parameters that can be used
        in several protocols definitions.
        Params:
            form: the Definition instance.
        """
        
        form.addParam('source', EnumParam,
                      label='Mask source',
                      default=SOURCE_GEOMETRY, choices=['Geometry','Created mask'], 
                      help='Select which type of mask do you want to apply. \n ')
        
        form.addParam('inputMask', PointerParam, pointerClass=self.MASK_CLASSNAME, 
                      condition='source==%d' % SOURCE_MASK,
                      label="Input mask")
        args={}
        args['isGeometry'] = 'source==%d' % SOURCE_GEOMETRY
        args['addSize'] = False
        if isinstance (self, XmippProtCreateMask3D):
            args['isFeatureFile'] = 'source==%d' % SOURCE_FEATURE_FILE
        self.GEOMETRY_BASECLASS.defineParams(self, form, **(args))
        
        form.addParam('fillType', EnumParam, 
                      choices=['value', 'min', 'max', 'avg'], 
                      condition='source==%d' % SOURCE_GEOMETRY,
                      default=MASK_FILL_VALUE,
                      label="Fill with ", display=EnumParam.DISPLAY_COMBO,
                      help='Select how are you going to fill the pixel values outside the mask. ')
        
        form.addParam('fillValue', IntParam, default=0, 
                      condition='fillType == %d and source==%d' % (MASK_FILL_VALUE,SOURCE_GEOMETRY),
                      label='Fill value',
                      help='Value to fill the pixel values outside the mask. ')
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertProcessStep(self):
        inputFn = self.inputFn

        if self.source == SOURCE_MASK:
            inputMaskFile = getImageLocation(self.inputMask.get())
            outputMaskFile = self._getTmpPath(self.MASK_FILE)
            self._insertFunctionStep('copyMaskFileStep', inputMaskFile, outputMaskFile)
        
        if self.fillType == MASK_FILL_VALUE:
            fillValue = self.fillValue.get()
        else:
            fillValue = self.getEnumText('fillType')
        
        if self.source == SOURCE_GEOMETRY:
            self._program = "xmipp_transform_mask"
            self._args += self._getGeometryCommand()
            self._args += " --substitute %(fillValue)s "
        elif self.source == SOURCE_MASK:
            self._program = "xmipp_image_operate"
            self._args += (" --mult %s" % outputMaskFile)
        else:
            raise Exception("Unrecognized mask type: %d" % self.source)
        
        self._insertFunctionStep("maskStep", self._args % locals())
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def copyMaskFileStep(self, inputMaskFile, outputMaskFile):
        """ Create a local copy of the mask.
        We use ImageHandler.convert instead of copyFile 
        because the mask could be inside an stack.
        """
        ImageHandler().convert(self.inputMask.get(), outputMaskFile)
    
    def maskStep(self, args):
        args += self._getExtraMaskArgs()
        self.runJob(self._program, args)            
        
    #--------------------------- UTILS functions ---------------------------------------------------
    def _getExtraMaskArgs(self):
        """ Return some extra arguments for the mask program.
        This function will varies when masking particles or volumes.
        """
        args = " -o %s --save_metadata_stack %s --keep_input_columns" % (self.outputStk, self.outputMd)
        return args

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self, geoClass):
        messages = []      
        messages.append("*Mask application*")
        if self.source.get() == SOURCE_GEOMETRY:
            messages.append(' Used geometrical mask:')
            messages += geoClass.summary(self)
        else:
            messages.append(' Used created mask: %s' % self.inputMask.get().getNameId())
            
        if self.fillType.get() == MASK_FILL_VALUE:
            messages.append(' Filled with %s value' % self.fillValue.get())
        else:
            messages.append(' Filled with %s value' % self.getEnumText('fillType'))
                                 
        return messages    

    def _methods(self, geoClass):
        messages = []

        if self.inputMask.get() is None:
            return messages

        messages.append("*Mask application*")
        
        if self.source.get() == SOURCE_GEOMETRY:
            messages.append("We applied a geometrical mask:")
            messages+=geoClass.methods(self)
        else:
            messages.append('We applied a created mask: %s' % self.inputMask.get().getNameId())
            
        if self.fillType.get() == MASK_FILL_VALUE:
            messages.append('filled with %s value' % self.fillValue.get())
        else:
            messages.append('filled with %s value' % self.getEnumText('fillType'))

        return messages
    
    def _validateDimensions(self, inputSetName, inputMaskName, errorMsg):
        """ Validate that input set (either particles or volumens) have the same
        dimension than the input mask.
        """
        errors = []
        
        if self.source == SOURCE_MASK:
            px, py, _ = self.getAttributeValue(inputSetName).getDim()
            mx, my, _ = self.getAttributeValue(inputMaskName).getDim()
            if px != mx or py != my:
                errors.append(errorMsg)
                
        return errors        
        
        
class XmippProtMaskParticles(ProtMaskParticles, XmippProcessParticles,
                             XmippProtMask, XmippGeometricalMask2D):
    """ Applies masks to a set of particle images to isolate specific regions
    or exclude unwanted background. The mask can be done with a geometry or
    import an external one. The center can be shift and the pixels outside the
    mask can be filled with a min avg, max or an specific value. Masking is
    critical for focusing on structural features and improving alignment and
    classification.

    AI Generated

    ## Overview

    The Apply 2D Mask protocol applies a mask to a set of particle images.

    Masking is commonly used in cryo-EM to isolate the particle region, suppress
    background, remove unwanted image areas, or focus processing on a region of
    interest. This protocol can apply either a geometrical mask created directly
    from user-defined parameters, or an already existing 2D mask.

    Pixels outside the mask can be replaced by a fixed value or by a statistic of
    the image, such as the minimum, maximum, or average value.

    The main output is a new particle set containing the masked particle images.

    ## Inputs and General Workflow

    The input is a set of particles.

    The user selects the mask source:

    - a geometrical mask;
    - a previously created mask.

    If a geometrical mask is selected, the protocol creates the mask internally
    using the particle box size and the selected geometrical parameters. If a
    created mask is selected, the protocol copies the mask to the protocol working
    area and applies it to the particles.

    The protocol then writes a new particle stack and metadata file, preserving the
    input metadata columns when possible.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be masked.

    The mask is applied independently to each particle image. The protocol does not
    change particle coordinates, orientations, CTF information, or alignment
    metadata. It only changes the image values according to the selected mask.

    The output particles can be used in downstream processing steps such as
    classification, alignment, reconstruction, or visualization.

    ## Mask Source

    The **Mask source** parameter controls how the mask is provided.

    There are two options:

    **Geometry** creates a mask from user-defined geometrical parameters.

    **Created mask** applies a mask that already exists as a Scipion mask object.

    Use a geometrical mask when a simple shape is sufficient. Use a created mask
    when the mask was generated previously, for example from another protocol or by
    manual editing.

    ## Geometrical Mask

    When **Geometry** is selected, the protocol creates the mask from the particle
    box size.

    The available 2D geometrical masks are:

    - Circular;
    - Box;
    - Crown;
    - Gaussian;
    - Raised cosine;
    - Raised crown.

    The mask is applied using `xmipp_transform_mask`.

    The geometrical mask is not saved as a separate output mask by this protocol;
    it is used internally to produce the masked particles.

    ## Circular Mask

    A **Circular** mask keeps a disk-shaped region.

    The **Radius** parameter defines the radius in pixels. If the radius is set to
    **-1**, the protocol uses half of the particle box size.

    Circular masks are useful for centered particles whose density lies mainly
    inside a round support.

    ## Box Mask

    A **Box** mask keeps a rectangular region.

    The **Box size** parameter defines the size of the box in pixels. If the value
    is **-1**, the protocol uses half of the particle box size.

    This option is useful for keeping a central square or rectangular region.

    ## Crown Mask

    A **Crown** mask keeps an annular region between an inner and an outer radius.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    If the outer radius is **-1**, the protocol uses half of the particle box size.

    Crown masks are useful when the user wants to keep ring-like information while
    excluding the center.

    ## Gaussian Mask

    A **Gaussian** mask applies a smooth Gaussian weighting.

    The **Sigma** parameter defines the Gaussian width in pixels. If sigma is set
    to **-1**, the protocol uses one sixth of the particle box size.

    Gaussian masks are useful when the user wants smooth attenuation rather than a
    hard binary boundary.

    ## Raised Cosine Mask

    A **Raised cosine** mask creates a smooth radial transition between an inner
    and an outer radius.

    This is useful when the user wants to reduce sharp-edge artifacts caused by
    hard masking.

    ## Raised Crown Mask

    A **Raised crown** mask creates a crown-like mask with smooth transitions at
    the borders.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**;
    - **Border decay**.

    The border decay controls the falloff at the crown boundaries.

    ## Shift Center

    The **Shift Center** option moves the center of the geometrical mask away from
    the center of the particle box.

    When enabled, the user provides:

    - **X center offset**;
    - **Y center offset**.

    This is useful when the region of interest is not centered in the particle
    image.

    ## Created Mask

    When **Created mask** is selected, the **Input mask** parameter defines the 2D
    mask to apply.

    The mask is copied locally and multiplied by the input particle images.

    The mask must have the same X and Y dimensions as the input particles. The
    protocol validates this requirement and reports an error if the dimensions do
    not match.

    ## Fill Type

    The **Fill with** parameter controls how pixels outside a geometrical mask are
    replaced.

    The available options are:

    - **value**;
    - **min**;
    - **max**;
    - **avg**.

    If **value** is selected, the user provides an explicit fill value. If one of
    the statistical options is selected, the protocol fills the outside region with
    the corresponding image statistic.

    This option is available for geometrical masks.

    ## Fill Value

    The **Fill value** parameter is used when **Fill with = value**.

    It defines the numerical value assigned to pixels outside the mask.

    A common choice is 0, which suppresses the background outside the selected
    region. Other values may be useful when the image background has a nonzero
    mean.

    ## Output Particles

    The main output is the masked particle set.

    The output images are written to a new stack, and the output metadata preserves
    the input columns when possible.

    The output particle set can be used directly in later Scipion protocols.

    ## Validation Rules

    When a created mask is used, the input particles and mask must have the same
    image dimensions.

    If the mask dimensions do not match the particle dimensions, the protocol
    reports a validation error.

    For geometrical masks, the mask size is derived from the particle dimensions.

    ## Interpreting the Result

    The output particles should be interpreted as the original particles after
    masking.

    Masking can improve focus on the particle signal and reduce background
    contribution, but it can also remove useful signal if the mask is too tight or
    miscentered.

    Sharp mask edges can introduce Fourier artifacts. Smooth masks such as
    Gaussian, raised cosine, or raised crown masks may be preferable for
    frequency-sensitive downstream steps.

    ## Practical Recommendations

    Use a circular or raised-cosine mask for centered particles.

    Use a shifted geometrical mask only when the region of interest is known to be
    off-center.

    Use an existing created mask when the mask was carefully generated from the
    data or manually curated.

    Make sure the mask is not too tight around the particle density.

    Use smooth masks when the output will be used for alignment, classification, or
    Fourier-space processing.

    Inspect representative masked particles before continuing downstream.

    ## Final Perspective

    Apply 2D Mask is a particle-image masking protocol.

    For biological users, its value is that it suppresses irrelevant image regions
    and focuses downstream analysis on the selected particle or region of interest.

    The protocol should be used carefully: masking improves many workflows, but an
    incorrect mask can remove real signal or introduce artifacts.
    """
    _label = 'apply 2d mask'
    
    MASK_FILE = 'mask.spi'
    MASK_CLASSNAME = 'Mask'
    GEOMETRY_BASECLASS = XmippGeometricalMask2D
    
    def __init__(self, **kwargs):
        ProtMaskParticles.__init__(self, **kwargs)
        XmippProcessParticles.__init__(self, **kwargs)
        XmippProtMask.__init__(self, **kwargs)
        self.allowThreads = False
        self.allowMpi = False

        
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippProtMask._defineProcessParams(self, form)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _getGeometryCommand(self):
        Xdim = self.inputParticles.get().getDimensions()[0]
        self.ndim = self.inputParticles.get().getSize()
        args = XmippGeometricalMask2D.argsForTransformMask(self, Xdim)
        return args

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        messages = []      
        messages += XmippProtMask._summary(self, XmippGeometricalMask2D)

        return messages

    def _methods(self):
        messages = []      
        messages += XmippProtMask._methods(self, XmippGeometricalMask2D)

        return messages
    
    def _validate(self):
        return XmippProtMask._validateDimensions(self, 
                                                 'inputParticles', 'inputMask',
                                                 'Input particles and mask should '
                                                 'have same dimensions.')
    
    
class XmippProtMaskVolumes(ProtMaskVolumes, XmippProcessVolumes, XmippProtMask, XmippGeometricalMask3D):
    """ Apply mask to a volume.

    AI Generated

    ## Overview

    The Apply 3D Mask protocol applies a mask to one volume or to a set of volumes.

    3D masking is used in cryo-EM to isolate molecular density, suppress solvent,
    focus on a region of interest, or prepare maps for alignment, refinement,
    subtraction, validation, filtering, or visualization. This protocol can apply
    either a geometrical 3D mask or an already created volume mask.

    For geometrical masks, voxels outside the mask can be filled with a fixed
    value or with a statistic of the volume, such as the minimum, maximum, or
    average value. For created masks, the protocol multiplies the input volume by
    the mask.

    The main output is a masked volume or a masked set of volumes.

    ## Inputs and General Workflow

    The input can be a single volume or a set of volumes.

    The user selects the mask source:

    - a geometrical mask;
    - a created 3D mask.

    If a geometrical mask is selected, the protocol creates the shape internally
    using the input volume dimensions. If a created mask is selected, the mask is
    copied locally and multiplied by the input volume or volumes.

    For a single input volume, the protocol writes one masked output map. For a set
    of volumes, it writes a new output volume set and preserves the input metadata
    when possible.

    ## Input Volumes

    The **Input volumes** parameter defines the map or maps to be masked.

    The protocol applies the mask to voxel values. It does not align, filter,
    resize, sharpen, or validate the maps.

    The output should be interpreted as the same input volume data restricted or
    modified according to the selected mask.

    ## Mask Source

    The **Mask source** parameter controls how the mask is provided.

    There are two options:

    **Geometry** creates a geometrical mask from user-defined parameters.

    **Created mask** applies an existing Scipion **VolumeMask** object.

    Use geometry for simple masks such as spheres, cylinders, boxes, or smooth
    radial masks. Use a created mask when the mask has been derived from the map,
    segmented, manually edited, or generated by another protocol.

    ## Geometrical Mask

    When **Geometry** is selected, the protocol creates the mask from the input
    volume size.

    The available 3D geometrical masks are:

    - Sphere;
    - Box;
    - Crown;
    - Cylinder;
    - Gaussian;
    - Raised cosine;
    - Raised crown.

    The mask is applied using `xmipp_transform_mask`.

    The geometrical mask is used internally and is not registered as a separate
    output mask by this protocol.

    ## Sphere Mask

    A **Sphere** mask keeps a spherical region.

    The **Radius** parameter defines the radius in pixels. If the value is **-1**,
    the protocol uses half of the input volume box size.

    Sphere masks are useful for approximately globular particles or simple central
    support regions.

    ## Box Mask

    A **Box** mask keeps a rectangular or cubic region.

    The **Box size** parameter defines the box size in pixels. If the value is
    **-1**, the protocol uses half of the input volume box size.

    This mask can be useful when the density region is approximately rectangular
    or when a central cubic region should be retained.

    ## Crown Mask

    A **Crown** mask keeps a shell-like region between an inner and an outer
    radius.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    If the outer radius is **-1**, the protocol uses half of the input volume box
    size.

    Crown masks are useful for selecting radial shells while excluding the center.

    ## Cylinder Mask

    A **Cylinder** mask keeps a cylindrical region.

    The relevant parameters are:

    - **Radius**;
    - **Height**.

    If the radius is **-1**, the protocol uses half of the input volume box size.
    If the height is **-1**, the protocol uses the full box size.

    Cylinder masks are useful for elongated structures, filaments, or simple
    helical support regions.

    ## Gaussian Mask

    A **Gaussian** mask applies a smooth 3D Gaussian weighting.

    The **Sigma** parameter defines the Gaussian width in pixels. If sigma is
    **-1**, the protocol uses one sixth of the input volume box size.

    Gaussian masks are useful when the user wants smooth attenuation rather than a
    binary boundary.

    ## Raised Cosine Mask

    A **Raised cosine** mask creates a smooth radial transition between an inner
    and an outer radius.

    This is useful for reducing artifacts caused by sharp mask boundaries.

    ## Raised Crown Mask

    A **Raised crown** mask creates a crown-like mask with smooth transitions at
    the inner and outer borders.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**;
    - **Border decay**.

    The border decay controls the falloff at the crown boundaries.

    ## Shift Center

    The **Shift center of the mask?** option moves the center of a geometrical mask
    away from the center of the input volume.

    When enabled, the user provides:

    - **X**;
    - **Y**;
    - **Z** center offsets.

    This is useful when the molecular region or region of interest is not centered
    in the volume.

    ## Created Mask

    When **Created mask** is selected, the **Input mask** parameter defines the
    VolumeMask to apply.

    The mask is copied locally and multiplied by the input volume or volume set.

    The mask must have the same X and Y dimensions as the input volumes. The
    protocol validates these dimensions. In practice, users should ensure that the
    full 3D dimensions, sampling rate, origin, and coordinate frame are compatible.

    ## Fill Type

    The **Fill with** parameter controls how voxels outside a geometrical mask are
    replaced.

    The available options are:

    - **value**;
    - **min**;
    - **max**;
    - **avg**.

    If **value** is selected, the user provides an explicit fill value. Otherwise,
    outside voxels are filled using the selected statistic.

    This option applies to geometrical masks.

    ## Fill Value

    The **Fill value** parameter is used when **Fill with = value**.

    It defines the numerical value assigned outside the geometrical mask.

    A common value is 0, which removes density outside the mask. Other values may
    be useful when the surrounding background should match a specific level.

    ## Output Volume

    If the input is a single volume, the output is a single masked volume.

    The output contains the result of applying the selected mask to the input map.

    This output can be used for visualization, alignment, focused processing,
    subtraction, validation, or other downstream workflows.

    ## Output Volume Set

    If the input is a set of volumes, the output is a masked volume set.

    Each output item corresponds to the matching input volume after mask
    application. Input metadata are preserved when possible.

    This is useful when the same mask should be applied consistently to several
    maps or class volumes.

    ## Validation Rules

    When a created mask is used, the input volumes and mask must have matching
    dimensions.

    If the dimensions do not match, the protocol reports a validation error.

    For geometrical masks, the mask size is derived from the input volume or volume
    set dimensions.

    ## Interpreting the Result

    The masked output should be interpreted as the original volume restricted or
    modified by the selected mask.

    Masking can improve visualization and reduce background influence. However, an
    incorrect mask can remove real density, create artificial boundaries, or bias
    downstream processing.

    Sharp masks can introduce Fourier artifacts. Smooth geometrical masks or
    carefully softened created masks are preferable when the output will be used in
    Fourier-space operations.

    ## Practical Recommendations

    Use an existing created mask when a molecule-specific mask has already been
    generated.

    Use a geometrical mask for simple support regions or quick preprocessing.

    Use smooth masks when the masked volume will be used for refinement,
    Fourier-space filtering, FSC analysis, or subtraction.

    Make sure the mask is not too tight around the density.

    Check that the mask and volume share the same size, sampling, origin, and
    coordinate frame.

    Inspect the masked map visually before using it in downstream workflows.

    ## Final Perspective

    Apply 3D Mask is a general volume-masking protocol.

    For biological users, its value is that it restricts 3D density to a selected
    region, reducing background and enabling focused analysis. It can be applied
    to one map or consistently to a set of maps.

    The protocol should be used with care because masks influence many downstream
    cryo-EM analyses. A good mask clarifies the region of interest; a poor mask can
    bias interpretation or processing.
    """
    _label = 'apply 3d mask'
    
    MASK_FILE = 'mask.vol'
    MASK_CLASSNAME = 'VolumeMask'
    GEOMETRY_BASECLASS = XmippGeometricalMask3D
    
    def __init__(self, **kwargs):
        ProtMaskVolumes.__init__(self, **kwargs)
        XmippProcessVolumes.__init__(self, **kwargs)
        XmippProtMask.__init__(self, **kwargs)
        self.allowMpi = False
        self.allowThreads = False
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippProtMask._defineProcessParams(self, form)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _getGeometryCommand(self):
        if self._isSingleInput():            
            Xdim = self.inputVolumes.get().getDim()[0]
        else:
            Xdim = self.inputVolumes.get().getDimensions()[0]
        args = XmippGeometricalMask3D.argsForTransformMask(self, Xdim)
        return args

    def _getExtraMaskArgs(self):
        if self._isSingleInput():
            return " -o %s" % self.outputStk
        else:
            return XmippProtMask._getExtraMaskArgs(self)

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        messages = []      
        messages += XmippProtMask._summary(self, XmippGeometricalMask3D)

        return messages
    
    def _methods(self):
        messages = []      
        messages += XmippProtMask._methods(self, XmippGeometricalMask3D)

        return messages
    
    def _validate(self):
        return XmippProtMask._validateDimensions(self, 
                                                 'inputVolumes', 'inputMask',
                                                 'Input volumes and mask should '
                                                 'have same dimensions.')
        
