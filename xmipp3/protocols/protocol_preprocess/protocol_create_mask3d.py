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
