# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#                J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

from pwem import emlib
from pwem.emlib.image import ImageHandler

from pyworkflow.protocol.params import PointerParam, StringParam, PathParam

from pwem.objects import VolumeMask
from pwem.protocols import ProtCreateMask3D


from xmipp3.convert import getImageLocation
from .geometrical_mask import *


SOURCE_VOLUME=0
SOURCE_GEOMETRY=1
SOURCE_FEATURE_FILE=2

OPERATION_THRESHOLD=0
OPERATION_SEGMENT=1
OPERATION_POSTPROCESS=2

SEGMENTATION_VOXELS=0
SEGMENTATION_AMINOACIDS=1
SEGMENTATION_DALTON=2
SEGMENTATION_AUTOMATIC=3

MORPHOLOGY_DILATION=0
MORPHOLOGY_EROSION=1
MORPHOLOGY_CLOSING=2
MORPHOLOGY_OPENING=3


class XmippProtCreateMask3D(ProtCreateMask3D, XmippGeometricalMask3D):
    """ Create a 3D mask.
    The mask can be created with a given geometrical shape (Sphere, Box,
    Cylinder...) or it can be obtained from operating on a 3d volume or a
    previous mask.

    AI Generated

    ## Overview

    The Create 3D Mask protocol generates a three-dimensional mask.

    3D masks are used in cryo-EM workflows to define the molecular region, remove
    background, restrict refinement or validation to a region of interest, create
    soft boundaries, or prepare maps for local processing. This protocol can create
    a mask from several sources:

    - an input volume;
    - a geometrical shape;
    - a feature file.

    After the initial mask is created, the protocol can optionally post-process it
    by removing small components, keeping only the largest component, applying
    symmetry, applying morphological operations, inverting the mask, or smoothing
    its borders.

    The main output is a Scipion 3D volume mask.

    ## Inputs and General Workflow

    The protocol first creates an initial mask according to the selected source.

    If the source is **Volume**, the mask is created from an input map by
    thresholding, segmentation, or direct post-processing.

    If the source is **Geometry**, the mask is created from a user-defined
    geometrical shape such as a sphere, box, crown, cylinder, Gaussian, raised
    cosine, or raised crown.

    If the source is **Feature File**, the mask is generated from an Xmipp phantom
    feature file.

    After creation, optional post-processing operations are applied. Finally, the
    mask is registered in Scipion with the appropriate sampling rate.

    ## Mask Source

    The **Mask source** parameter defines how the initial mask is created.

    There are three options:

    **Volume** creates the mask from an existing 3D volume.

    **Geometry** creates the mask from a user-defined geometrical shape.

    **Feature File** creates the mask from a phantom feature file describing one or
    more shapes.

    The best option depends on the use case. Volume-derived masks are useful when
    the mask should follow density. Geometrical masks are useful for simple,
    reproducible shapes. Feature files are useful for more complex synthetic masks.

    ## Creating a Mask from a Volume

    When **Volume** is selected as the source, the **Input volume** parameter
    defines the map used to create the mask.

    The protocol supports three operations:

    - **Threshold**;
    - **Segment**;
    - **Only postprocess**.

    The sampling rate of the output mask is copied from the input volume.

    This mode is useful when the user wants the mask to follow the density present
    in an experimental or processed map.

    ## Threshold Operation

    The **Threshold** operation creates a binary mask from the input volume.

    The **Threshold** parameter defines the gray-value cutoff. Voxels with values
    below the threshold are set to 0, and the remaining voxels become part of the
    mask.

    This is a simple and common way to create a mask from density.

    The threshold should be chosen carefully. If it is too high, weak but real
    density may be excluded. If it is too low, background noise may be included.

    ## Segmentation Operation

    The **Segment** operation creates a mask by selecting a region corresponding to
    a desired size or mass.

    The available segmentation types are:

    **Number of voxels**, where the user specifies the target number of voxels.

    **Number of aminoacids**, where the user specifies the approximate number of
    amino acids and the protocol uses the sampling rate to estimate the mask.

    **Dalton mass**, where the user specifies the molecular mass in daltons.

    **Automatic**, where the protocol uses Otsu thresholding.

    Segmentation is useful when the expected molecular size is known and the user
    wants a mask that reflects that approximate amount of density.

    ## Only Postprocess Operation

    The **Only postprocess** operation copies the input volume as the starting
    mask.

    No thresholding or segmentation is applied at the creation step. The selected
    post-processing operations are then applied to the copied volume.

    This option is useful when the input is already a mask, or when the user wants
    to apply additional post-processing to an existing mask-like volume.

    ## Creating a Mask from Geometry

    When **Geometry** is selected as the source, the protocol creates a mask from a
    geometrical shape.

    The user defines the mask size, sampling rate, shape, and shape-specific
    parameters. The output mask is cubic, with dimensions:

    \[
    \text{size} \times \text{size} \times \text{size}
    \]

    This mode does not require an input volume.

    ## Sampling Rate for Geometry or Feature File

    The **Sampling Rate** parameter is used when the mask source is **Geometry** or
    **Feature File**.

    It defines the physical voxel size of the generated mask, in angstroms per
    pixel.

    This value should match the volume or particles that will later use the mask.

    ## Mask Size

    The **Mask size** parameter is used for geometrical masks.

    It defines the cubic dimensions of the generated mask in voxels.

    For example, a size of 128 creates a 128 × 128 × 128 voxel mask.

    The mask size should match the box size of the maps where the mask will be
    used.

    ## Geometrical Mask Types

    The available 3D geometrical mask types are:

    - Sphere;
    - Box;
    - Crown;
    - Cylinder;
    - Gaussian;
    - Raised cosine;
    - Raised crown.

    Each type exposes its own parameters. These parameters are expressed in pixels
    or voxels.

    ## Sphere Mask

    A **Sphere** mask creates a spherical region.

    The **Radius** parameter defines the sphere radius in pixels. If the radius is
    set to **-1**, the protocol uses half the mask size.

    Sphere masks are useful for approximately globular particles or for simple
    central support regions.

    ## Box Mask

    A **Box** mask creates a cubic or rectangular support region.

    The **Box size** parameter defines the box size. If it is set to **-1**, the
    protocol uses half the mask size.

    This mask can be useful when the region of interest is approximately
    rectangular or when the user wants to keep a central cubic region.

    ## Crown Mask

    A **Crown** mask creates a shell-like region between an inner and an outer
    radius.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    If the outer radius is set to **-1**, the protocol uses half the mask size.

    Crown masks are useful for selecting a radial shell while excluding the central
    region.

    ## Cylinder Mask

    A **Cylinder** mask creates a cylindrical region.

    The relevant parameters are:

    - **Radius**;
    - **Height**.

    If the radius is set to **-1**, the protocol uses half the mask size. If the
    height is set to **-1**, the protocol uses the full mask size.

    Cylinder masks are useful for elongated particles, filaments, or approximate
    helical support regions.

    ## Gaussian Mask

    A **Gaussian** mask creates a smooth three-dimensional Gaussian weighting
    function.

    The **Sigma** parameter defines the Gaussian width in pixels. If sigma is set
    to **-1**, the protocol uses one sixth of the mask size.

    Gaussian masks are useful when the user wants smooth weighting rather than a
    hard binary support.

    ## Raised Cosine Mask

    A **Raised cosine** mask creates a smooth radial transition between an inner
    and an outer radius.

    This is useful when the user wants to avoid sharp mask edges, which can
    introduce Fourier artifacts.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    ## Raised Crown Mask

    A **Raised crown** mask creates a crown-like mask with smooth transitions at
    both borders.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**;
    - **Border decay**.

    The border decay controls the falloff at the crown boundaries.

    ## Shift Center

    The **Shift center of the mask?** option allows the user to move the center of
    a geometrical mask away from the center of the box.

    When enabled, the user provides X, Y, and Z center offsets in pixels.

    This is useful when the region of interest is not centered in the volume.

    ## Creating a Mask from a Feature File

    When **Feature File** is selected, the protocol creates the mask using an
    Xmipp phantom feature file.

    The feature file describes one or more geometrical features, their operations,
    densities, centers, and shape-specific parameters. Examples of supported
    feature types include spheres, blobs, Gaussians, cylinders, double cylinders,
    cubes, ellipsoids, and cones.

    This option is useful for complex synthetic masks that are easier to describe
    in a feature file than through the graphical form.

    ## Remove Small Objects

    The **Remove small objects** option removes connected components smaller than
    a selected size.

    The **Minimum size** parameter defines the minimum component size in voxels.

    This is useful for cleaning masks created from noisy thresholding or
    segmentation, where small isolated islands may appear outside the main
    structure.

    The input mask should be binary when using this option.

    ## Keep Largest Component

    The **Keep largest component** option keeps only the largest connected
    component of the mask.

    This is useful when a thresholded or segmented mask contains several separated
    regions, and the user wants to keep only the main molecular component.

    The input mask should be binary.

    ## Symmetrize Mask

    The **Symmetrize mask** option applies a selected symmetry group to the mask.

    The **Symmetry group** parameter defines the Xmipp symmetry, with **c1**
    meaning no symmetry.

    If the mask was created by thresholding or segmentation, the protocol binarizes
    it again after symmetrization.

    Symmetry should be used only when the biological object and the intended mask
    are expected to be symmetric.

    ## Morphological Operation

    The **Apply morphological operation** option applies a binary morphological
    operation to the mask.

    The available operations are:

    **Dilation**, which expands the white region.

    **Erosion**, which shrinks the white region.

    **Closing**, which performs dilation followed by erosion and can remove black
    holes or gaps.

    **Opening**, which performs erosion followed by dilation and can remove small
    white objects.

    The **Structural element size** controls the strength of the operation. Larger
    values produce stronger effects.

    ## Invert the Mask

    The **Invert the mask** option swaps the inside and outside of the mask.

    The protocol multiplies the mask by -1 and then adds 1, converting 1-valued
    regions to 0 and 0-valued regions to 1.

    This is useful when the user wants to select the complement of the current
    mask.

    ## Smooth Borders

    The **Smooth borders** option smooths the mask by convolving it with a
    Gaussian.

    The **Gaussian sigma** parameter defines the width of the smoothing kernel in
    pixels.

    After smoothing, negative values are clipped to zero.

    Smoothing is often useful for reducing sharp mask boundaries, which can create
    artifacts in Fourier-space operations.

    ## Output Mask

    The main output is **outputMask**.

    This output is a Scipion **VolumeMask** object pointing to the generated
    `mask.mrc` file.

    If the source is **Volume**, the output mask uses the sampling rate of the
    input volume. If the source is **Geometry** or **Feature File**, it uses the
    sampling rate provided by the user.

    The protocol also writes the sampling rate into the image header.

    ## Interpreting the Output

    The output should be interpreted as a 3D mask created by the selected source
    and post-processing operations.

    A mask created from a volume reflects density and thresholding or segmentation
    choices. A geometrical mask reflects user-defined shape parameters. A feature
    file mask reflects the external feature description.

    The mask may be binary or smoothly weighted, depending on the selected
    creation and post-processing operations.

    ## Practical Recommendations

    Use volume thresholding for simple density-derived masks.

    Use segmentation when the approximate molecular size or mass is known.

    Use geometrical masks when a simple reproducible shape is sufficient.

    Use feature files for complex synthetic masks.

    Remove small objects and keep the largest component when masks contain
    unwanted islands.

    Use smoothing before Fourier-space operations or refinement steps where sharp
    mask edges may cause artifacts.

    Set the sampling rate and box size to match the volume where the mask will be
    used.

    Inspect the mask visually before using it in downstream protocols.

    Avoid masks that are too tight, because they may remove real density. Avoid
    masks that are too loose, because they may include solvent and noise.

    ## Final Perspective

    Create 3D Mask is a flexible mask-generation and mask-processing protocol.

    For biological users, its value is that it can create masks from experimental
    density, from known geometrical shapes, or from feature-file descriptions, and
    then refine those masks with common post-processing operations.

    The resulting mask can be used in reconstruction, refinement, validation,
    local-resolution estimation, subtraction, sharpening, or visualization
    workflows where defining the molecular region is essential.
    """
    _label = 'create 3d mask'
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Mask generation')
        form.addParam('source', EnumParam, default=SOURCE_VOLUME,
                      choices=['Volume', 'Geometry', 'Feature File'	],
                      label='Mask source')
        # For volume sources
        isVolume = 'source==%d' % SOURCE_VOLUME
        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label="Input volume", allowsNull=True, condition=isVolume,
                      help="Select the volume that will be used to create the mask")
        form.addParam('volumeOperation', EnumParam, default=OPERATION_THRESHOLD,
                      choices=['Threshold', 'Segment', 'Only postprocess'],
                      label='Operation', condition=isVolume)
        #TODO: add wizard
        form.addParam('threshold', FloatParam, default=0.0,
                      condition='volumeOperation==%d and %s'
                      % (OPERATION_THRESHOLD, isVolume),
                      label='Threshold',
                      help="Select the threshold. Gray values lesser than the threshold" \
                           "will be set to zero, otherwise will be one (mask area).")
        isSegmentation = 'volumeOperation==%d and %s' % (OPERATION_SEGMENT, isVolume)
        form.addParam('segmentationType', EnumParam, default=SEGMENTATION_DALTON, 
                      condition=isSegmentation,
                      label='Segmentation type',
                      choices=['Number of voxels', 'Number of aminoacids',
                               'Dalton mass', 'Automatic'])
        form.addParam('nvoxels', IntParam, 
                      condition='%s and segmentationType==%d'
                      % (isSegmentation, SEGMENTATION_VOXELS),
                      label='Number of voxels')
        form.addParam('naminoacids', IntParam,
                      condition='%s and segmentationType==%d'
                      % (isSegmentation, SEGMENTATION_AMINOACIDS),
                      label='Number of aminoacids')
        form.addParam('dalton', FloatParam, 
                      condition='%s and segmentationType==%d'
                      % (isSegmentation, SEGMENTATION_DALTON),
                      label='Mass (Da)')
        
        # For geometrical sources
        form.addParam('samplingRate', FloatParam, default=1, 
                      condition='source==%d or source==%d' % (SOURCE_GEOMETRY, SOURCE_FEATURE_FILE), 
                      label="Sampling Rate (Å/px)")
        XmippGeometricalMask3D.defineParams(self, form, 
                                            isGeometry='source==%d'
                                            % SOURCE_GEOMETRY,
                                            addSize=True)
        # Feature File
        isFeatureFile = 'source==%d' % SOURCE_FEATURE_FILE
        form.addParam('featureFilePath', PathParam,
                      condition=isFeatureFile,
                      label="Feature File",
                      help="""Create a mask using a feature file. Follows an example of feature file 
# XMIPP_STAR_1 *
# Type of feature (sph, blo, gau, Cyl, dcy, cub, ell, con)(Required)
# The operation after adding the feature to the phantom (+/=) (Required)
# The feature density (Required)
# The feature center (Required)
# The vector for special parameters of each vector (Required)
# Sphere: [radius] 
# Blob : [radius alpha m] Gaussian : [sigma]
# Cylinder : [xradius yradius height rot tilt psi]
# DCylinder : [radius height separation rot tilt psi]
# Cube : [xdim ydim zdim rot tilt psi]
# Ellipsoid : [xradius yradius zradius rot tilt psi]
# Cone : [radius height rot tilt psi]
data_block1
 _dimensions3D  '34 34 34' 
 _phantomBGDensity  0.
 _scale  1.
data_block2
loop_
 _featureType
 _featureOperation
 _featureDensity
 _featureCenter
 _featureSpecificVector
sph + 1 '3.03623188  0.02318841 -5.04130435' '7'
"""
)

        # Postprocessing
        form.addSection(label='Postprocessing')
        form.addParam('doSmall', BooleanParam, default=False,
                      label='Remove small objects',
                      help="To remove small clusters of points. "
                           "The input mask has to be binary.")
        form.addParam('smallSize', IntParam, default=50,
                      label='Minimum size',condition="doSmall",
                      help='Connected components whose size is smaller than '
                           'this number in voxels will be removed')
        form.addParam('doBig', BooleanParam, default=False,
                      label='Keep largest component',
                      help="To keep cluster greater than a given size. The input mask has to be binary")
        form.addParam('doSymmetrize', BooleanParam, default=False,
                      label='Symmetrize mask')
        form.addParam('symmetry', StringParam, default='c1',
                      label='Symmetry group',condition="doSymmetrize",
                      help="To obtain a symmetric mask. See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry \n"
                           "for a description of the symmetry groups format. \n"
                           "If no symmetry is present, give c1")
        form.addParam('doMorphological', BooleanParam, default=False,
                      label='Apply morphological operation',
                      help="Dilation (dilate white region). \n"
                           "Erosion (erode white region). \n"
                           "Closing (Dilation+Erosion, removes black spots). \n"
                           "Opening (Erosion+Dilation, removes white spots). \n")
        form.addParam('morphologicalOperation', EnumParam, default=MORPHOLOGY_DILATION,
                      condition="doMorphological", 
                      choices=['dilation', 'erosion', 'closing', 'opening'],
                      label='Operation')
        form.addParam('elementSize', IntParam, default=1, condition="doMorphological",
                      label='Structural element size',                      
                      help="The larger this value, the more the effect will be noticed")
        form.addParam('doInvert', BooleanParam, default=False,
                      label='Invert the mask')
        form.addParam('doSmooth', BooleanParam, default=False,
                      label='Smooth borders',
                      help="Smoothing is performed by convolving the mask with a Gaussian.")
        form.addParam('sigmaConvolution', FloatParam, default=2, condition="doSmooth",
                      label='Gaussian sigma (px)',
                      help="The larger this value, the more the effect will be noticed")

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.maskFile = self._getPath('mask.mrc')
        
        if self.source == SOURCE_VOLUME:
            self._insertFunctionStep(self.createMaskFromVolumeStep)
        elif self.source == SOURCE_GEOMETRY:
            self.inputVolume.set(None)
            self._insertFunctionStep(self.createMaskFromGeometryStep)
        elif self.source == SOURCE_FEATURE_FILE:
            self._insertFunctionStep(self.createMaskFromFeatureFile)
        self._insertFunctionStep(self.postProcessMaskStep)
        self._insertFunctionStep(self.createOutputStep)
    
    #--------------------------- STEPS functions -------------------------------
    def createMaskFromVolumeStep(self):
        volume = self.inputVolume.get()
        fnVol = getImageLocation(volume)
        if fnVol.endswith(".mrc"):
            fnVol += ":mrc"
        Ts = volume.getSamplingRate()
        
        if self.volumeOperation == OPERATION_THRESHOLD:
            self.runJob("xmipp_transform_threshold",
                        "-i %s -o %s --select below %f --substitute binarize"
                        % (fnVol, self.maskFile, self.threshold.get()))
        elif self.volumeOperation == OPERATION_SEGMENT:
            args="-i %s -o %s --method " % (fnVol, self.maskFile)
            if self.segmentationType == SEGMENTATION_VOXELS:
                args += "voxel_mass %d" % (self.nvoxels.get())
            elif self.segmentationType == SEGMENTATION_AMINOACIDS:
                args += "aa_mass %d %f" % (self.naminoacids.get(), Ts)
            elif self.segmentationType == SEGMENTATION_DALTON:
                args += "dalton_mass %d %f" % (self.dalton.get(), Ts)
            else:
                args += "otsu"
            self.runJob("xmipp_volume_segment", args)
        
        elif self.volumeOperation == OPERATION_POSTPROCESS:
            ImageHandler().convert(fnVol,self.maskFile)
            
        return [self.maskFile]
        
    def createMaskFromGeometryStep(self):
        # Create empty volume file with desired dimensions
        size = self.size.get()
        emlib.createEmptyFile(self.maskFile, size, size, size)
        
        # Create the mask
        args = '-i %s ' % self.maskFile
        args += XmippGeometricalMask3D.argsForTransformMask(self,size)
        args += ' --create_mask %s' % self.maskFile
        self.runJob("xmipp_transform_mask", args)
        
        return [self.maskFile]

    def createMaskFromFeatureFile(self):
        featFileName = self.featureFilePath.get()
        
        self.runJob("xmipp_phantom_create", "-i %s -o %s"
                % (featFileName, self.maskFile) )
            
        return [self.maskFile]

    def postProcessMaskStep(self):        
        if self.doSmall:
            self.runJob("xmipp_transform_morphology","-i %s --binaryOperation removeSmall %d"
                        % (self.maskFile,self.smallSize.get()))
        
        if self.doBig:
            self.runJob("xmipp_transform_morphology","-i %s --binaryOperation keepBiggest"
                        % self.maskFile)
        
        if self.doSymmetrize:
            if self.symmetry!='c1':
                self.runJob("xmipp_transform_symmetrize","-i %s --sym %s --dont_wrap"%(self.maskFile,self.symmetry.get()))
                if self.volumeOperation==OPERATION_THRESHOLD or self.volumeOperation==OPERATION_SEGMENT:
                    self.runJob("xmipp_transform_threshold",
                                "-i %s --select below 0.5 --substitute binarize" % self.maskFile)
        
        if self.doMorphological:
            self.runJob("xmipp_transform_morphology","-i %s --binaryOperation %s --size %d"
                        % (self.maskFile, self.getEnumText('morphologicalOperation'),
                           self.elementSize.get()))
        
        if self.doInvert:
            self.runJob("xmipp_image_operate","-i %s --mult -1"%self.maskFile)
            self.runJob("xmipp_image_operate","-i %s --plus  1"%self.maskFile)
        
        if self.doSmooth:
            self.runJob("xmipp_transform_filter","-i %s --fourier real_gaussian %f"%(self.maskFile,self.sigmaConvolution.get()))
            self.runJob("xmipp_transform_threshold",
                        "-i %s --select below 0 --substitute value 0" % self.maskFile)

    def createOutputStep(self):
        volMask = VolumeMask()
        volMask.setFileName(self.maskFile)
        
        if self.source==SOURCE_VOLUME:
            Ts = self.inputVolume.get().getSamplingRate()
        else:
            Ts = self.samplingRate.get()
        volMask.setSamplingRate(Ts)
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(self.maskFile,Ts))

        self._defineOutputs(outputMask=volMask)
        
        if self.source==SOURCE_VOLUME:
            self._defineSourceRelation(self.inputVolume, self.outputMask)
        
    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        messages = []      
        messages.append("*Mask creation*")
        if self.source==SOURCE_VOLUME:
            if self.volumeOperation==OPERATION_THRESHOLD:
                messages.append("   Thresholding %f"%self.threshold.get())
            elif self.volumeOperation==OPERATION_SEGMENT:
                if self.segmentationType==SEGMENTATION_AUTOMATIC:
                    messages.append("   Automatically segmented")
                else:
                    m="   Segmented to a mass of "
                    if self.segmentationType==SEGMENTATION_VOXELS:
                        m+="%d voxels"%(int(self.nvoxels.get()))
                    elif self.segmentationType==SEGMENTATION_AMINOACIDS:
                        m+="%d aminoacids"%(int(self.naminoacids.get()))
                    elif self.segmentationType==SEGMENTATION_DALTON:
                        m+="%d daltons"%(int(self.dalton.get()))
                    messages.append(m)
        elif self.source==SOURCE_GEOMETRY:
            size = self.size.get()
            messages.append("   Mask of size: %d x %d x %d"%(size,size,size))
            messages += XmippGeometricalMask3D.summary(self)

        messages.append("*Mask processing*")
        if self.doSmall:
            messages.append("   Removing components smaller than %d" % self.smallSize.get())
        if self.doBig:
            messages.append("   Keeping largest component")
        if self.doSymmetrize:
            messages.append("   Symmetrized %s" % self.symmetry.get())
        if self.doMorphological:
            messages.append("   Morphological operation: %s"
                            % self.getEnumText('morphologicalOperation'))
        if self.doInvert:
            messages.append("   Inverted")
        if self.doSmooth:
            messages.append("   Smoothed (sigma=%f)"%self.sigmaConvolution.get())
        return messages

    def _citations(self):
        if (self.source == SOURCE_VOLUME and 
            self.volumeOperation == OPERATION_SEGMENT and 
            self.segmentationType==SEGMENTATION_AUTOMATIC):
            return ['Otsu1979']

    def _methods(self):
        messages = []

        if self.inputVolume.get() is None:
            return messages

        messages.append("*Mask creation*")

        if self.source == SOURCE_VOLUME:
            messages.append('We processed the volume %s.'%self.inputVolume.get().getNameId())

            if self.volumeOperation == OPERATION_THRESHOLD:
                messages.append("We thresholded it to a gray value of %f. "%self.threshold.get())
            elif self.volumeOperation == OPERATION_SEGMENT:
                if self.segmentationType == SEGMENTATION_AUTOMATIC:
                    messages.append("We automatically segmented it using Otsu's method [Otsu1979]")
                else:
                    m="We segmented it to a mass of "
                    if self.segmentationType == SEGMENTATION_VOXELS:
                        m+="%d voxels"%(int(self.nvoxels.get()))
                    elif self.segmentationType == SEGMENTATION_AMINOACIDS:
                        m+="%d aminoacids"%(int(self.naminoacids.get()))
                    elif self.segmentationType == SEGMENTATION_DALTON:
                        m+="%d daltons"%(int(self.dalton.get()))
                    messages.append(m)
        
        elif self.source == SOURCE_GEOMETRY:
            size=self.size.get()
            messages.append("We created a mask of size: %d x %d x %d voxels. "
                            % (size, size, size))
            messages += XmippGeometricalMask3D.methods(self)

        if self.doSmall:
            messages.append("We removed components smaller than %d voxels."
                            % (self.smallSize.get()))
        if self.doBig:
            messages.append("We kept the largest component. ")
        if self.doSymmetrize:
            messages.append("We symmetrized it as %s. "%self.symmetry.get())
        if self.doMorphological:
            messages.append("Then, we applied a morphological operation, concisely, a %s. "
                            % self.getEnumText('morphologicalOperation'))
        if self.doInvert:
            messages.append("We inverted the mask. ")
        if self.doSmooth:
            messages.append("And, we smoothed it (sigma=%f voxels)."
                            % self.sigmaConvolution.get())
        if self.hasAttribute('outputMask'):
            messages.append('We refer to the output mask as %s.'  % self.outputMask.getNameId())
        return messages
    
    def _validate(self):
        errors = []
        if self.source == SOURCE_VOLUME and not self.inputVolume.get():
            errors.append("You need to select an input volume")
            return errors
