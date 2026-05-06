# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Laura del Cano (ldelcano@cnb.csic.es)
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
from pwem.objects import Mask
from pwem.protocols import ProtCreateMask2D

from .geometrical_mask import *


SOURCE_PARTICLE=0
SOURCE_GEOMETRY=1
SOURCE_MASK=2


class XmippProtCreateMask2D(ProtCreateMask2D, XmippGeometricalMask2D):
    """ Create a 2D mask.
    The mask can be created with a given geometrical shape (Circle, Rectangle,
    Crown...) or it can be obtained from operating on a 2d image or a previuous
    mask.

    AI Generated

    ## Overview

    The Create 2D Mask protocol generates a two-dimensional mask from a selected
    geometrical shape.

    Masks are frequently used in cryo-EM image processing to restrict operations
    to a region of interest and to suppress background. A 2D mask can be useful
    for particle images, class averages, projections, or other two-dimensional
    image-processing steps.

    This protocol creates a mask directly from geometry. The user defines the mask
    size, sampling rate, shape, and shape-specific parameters such as radius, box
    size, inner and outer radii, Gaussian sigma, or border decay.

    The main output is a Scipion mask object.

    ## Inputs and General Workflow

    The protocol does not require an input image.

    It creates an empty 2D image of the requested size and then applies the
    selected geometrical mask using the Xmipp mask transformation program. The
    result is saved as `mask.xmp`.

    Finally, the protocol registers this file as the output mask and assigns the
    sampling rate selected by the user.

    ## Sampling Rate

    The **Sampling Rate** parameter defines the pixel size of the output mask, in
    angstroms per pixel.

    This value does not change the shape of the mask in pixels. Instead, it
    defines the physical scale associated with the mask when it is used in later
    protocols.

    The sampling rate should match the images to which the mask will be applied.

    ## Mask Size

    The **Mask size** parameter defines the dimensions of the output mask in
    pixels.

    The output mask is a square image of size:

    \[
    \text{size} \times \text{size}
    \]

    For example, a value of 256 creates a 256 × 256 pixel mask.

    The mask size should match the box size of the images that will use the mask.

    ## Mask Type

    The **Mask type** parameter selects the geometrical shape of the mask.

    The available 2D mask types are:

    - Circular;
    - Box;
    - Crown;
    - Gaussian;
    - Raised cosine;
    - Raised crown.

    Each type exposes its own parameters. For example, circular masks use a radius,
    box masks use a box size, crown-like masks use inner and outer radii, and
    Gaussian masks use sigma.

    ## Circular Mask

    A **Circular** mask creates a disk-shaped region centered in the image, unless
    a center shift is requested.

    The **Radius** parameter defines the disk radius in pixels. If the radius is
    set to **-1**, the protocol uses half the mask size.

    Circular masks are useful when the object of interest is approximately
    centered and roughly round in projection.

    ## Box Mask

    A **Box** mask creates a rectangular or square region.

    The **Box size** parameter defines the size of the box in pixels. If it is set
    to **-1**, the protocol uses half the mask size.

    Box masks are useful when the region of interest is rectangular or when the
    user wants to keep a central square area.

    ## Crown Mask

    A **Crown** mask creates an annular region between an inner and an outer
    radius.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    If the outer radius is set to **-1**, the protocol uses half the mask size.

    Crown masks are useful when the user wants to select a ring-like region while
    excluding the central area.

    ## Gaussian Mask

    A **Gaussian** mask creates a smooth Gaussian weighting function.

    The **Sigma** parameter defines the Gaussian width in pixels. If sigma is set
    to **-1**, the protocol uses one sixth of the mask size.

    Gaussian masks are useful when the user wants a smooth weighting rather than a
    hard binary mask.

    ## Raised Cosine Mask

    A **Raised cosine** mask creates a smooth radial transition between an inner
    and an outer radius.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**.

    This mask is useful when a smooth falloff is desired, reducing sharp-edge
    artifacts that can appear with hard masks.

    ## Raised Crown Mask

    A **Raised crown** mask creates a crown-like mask with smooth transitions at
    the borders.

    The relevant parameters are:

    - **Inner radius**;
    - **Outer radius**;
    - **Border decay**.

    The border decay controls the falloff of the two crown borders.

    This mask is useful for ring-shaped regions where smooth edges are important.

    ## Shift Center

    The **Shift Center** option allows the user to move the center of the mask away
    from the image center.

    When enabled, the user provides:

    - **X center offset**;
    - **Y center offset**.

    This is useful when the region of interest is not centered in the image.

    The shift is expressed in pixels.

    ## Output Mask

    The main output is **outputMask**.

    This output points to the generated `mask.xmp` file and stores the sampling
    rate selected by the user.

    The mask can be used in later Scipion protocols that accept 2D masks.

    ## Interpreting the Output

    The output should be interpreted as a geometrical 2D mask.

    Depending on the selected type, the mask may be binary, smoothly weighted, or
    radially tapered. The protocol does not derive the mask from image content; it
    creates it entirely from user-defined geometry.

    Therefore, the correctness of the mask depends on choosing a size, shape,
    center, and radii that match the images where the mask will be applied.

    ## Practical Recommendations

    Set the mask size equal to the box size of the images that will use it.

    Set the sampling rate equal to the sampling rate of those images.

    Use a circular mask for centered particles or projections.

    Use raised cosine or raised crown masks when smooth edges are important.

    Use center shifting only when the region of interest is known to be off-center.

    Inspect the generated mask before using it in downstream processing.

    Avoid masks that are too tight, because they may remove real signal. Avoid
    masks that are too loose, because they may include unnecessary background.

    ## Final Perspective

    Create 2D Mask is a simple geometrical mask-generation protocol.

    For biological users, its value is practical: it creates a reproducible 2D mask
    with known size, sampling rate, shape, and center. Such masks can be used to
    focus later image-processing steps on the relevant particle or projection
    region while reducing the influence of background.
    """
    _label = 'create 2d mask'
    
    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Mask generation')
        
        # For geometrical sources
        form.addParam('samplingRate', FloatParam, default=1, 
                      label="Sampling Rate (Å/px)")
        
        XmippGeometricalMask2D.defineParams(self, form, 
                                            isGeometry=True, 
                                            addSize=True)


    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.maskFile = self._getPath('mask.xmp')
    
        self._insertFunctionStep('createMaskFromGeometryStep')

        self._insertFunctionStep('postProcessMaskStep')
        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions -------------------------------
        
    def createMaskFromGeometryStep(self):
        # Create empty volume file with desired dimensions
        size = self.size.get()
        emlib.createEmptyFile(self.maskFile, size, size)
        
        # Create the mask
        args = '-i %s ' % self.maskFile
        args += XmippGeometricalMask2D.argsForTransformMask(self,size)
        args += ' --create_mask %s' % self.maskFile
        self.runJob("xmipp_transform_mask", args)
        
        return [self.maskFile]


    def postProcessMaskStep(self):
        pass

    def createOutputStep(self):
        mask = Mask()
        mask.setFileName(self.maskFile)
        
        mask.setSamplingRate(self.samplingRate.get())
        
        self._defineOutputs(outputMask=mask)
        
    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        messages = []      
        messages.append("*Mask creation*")
        size = self.size.get()
        messages.append("   Mask of size: %d x %d"%(size,size))
        messages += XmippGeometricalMask2D.summary(self)

        return messages

    def _citations(self):
        pass

    def _methods(self):
        messages = []      
        messages.append("*Mask creation*")
        
        size=self.size.get()
        messages.append("We created a mask of size: %d x %d pixels. "%(size,size))
        messages+=XmippGeometricalMask2D.methods(self)

        if self.hasAttribute('outputMask'):
            messages.append('Output mask: %s.' % self.getObjectTag('outputMask'))
        return messages
    
