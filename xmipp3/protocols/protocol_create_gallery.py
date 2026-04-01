# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Federico P. de Isidro-Gomez
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
from pyworkflow import VERSION_1_1
from pyworkflow.protocol import PointerParam, StringParam, FloatParam, EnumParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow import BETA, UPDATED, NEW, PROD

from pwem.protocols import ProtAnalysis3D


from xmipp3.convert import readSetOfParticles


class XmippProtCreateGallery(ProtAnalysis3D):
    """
    Create a gallery of projections from a volume.
    This gallery of projections may help to understand the images
    observed in the microscope.

    AI Generated:

    ## Overview

    The Create Gallery protocol generates a systematic set of 2D projections
    from a 3D volume. These projections simulate how the structure would appear
    from different viewing directions, providing a direct link between the
    reconstructed volume and the particle images observed in the microscope.

    For biological users, this protocol is especially useful for interpreting
    2D class averages, validating reconstructions, or gaining intuition about
    the structural features of a macromolecule. By comparing experimental
    images with the generated projections, one can assess whether the
    reconstructed volume is consistent with the data and whether certain views
    are over- or under-represented.

    This type of gallery is also commonly used for visualization, teaching,
    and figure preparation.

    ## Inputs and General Workflow

    The protocol requires a **3D volume** as input. From this volume, it
    computes projections over a range of orientations defined by the user.

    The angular sampling is controlled by specifying ranges for **rotational
    (in-plane rotation)** and **tilt (out-of-plane angle)**. The protocol
    systematically samples these angles to generate a grid of projections
    covering the orientation space.

    If symmetry is present in the structure, it can be specified so that
    redundant views are avoided and the gallery reflects the true symmetry of
    the particle.

    The output is a set of 2D images representing projections of the volume
    from different directions, which can be directly compared with experimental
    particle images or class averages.

    ## Angular Sampling: Exploring the Orientation Space

    The most important parameters for biological interpretation are the angular
    ranges and steps.

    The **rotational angle** typically spans 0 to 360 degrees and controls
    rotation around the projection axis. The **tilt angle** determines the
    viewing direction, where 0 degrees corresponds to a top view and 90 degrees
    to a side view.

    The step size defines how densely the orientation space is sampled. Smaller
    steps produce a more detailed and complete gallery but increase
    computational cost and the number of generated images.

    From a practical point of view, coarse sampling is often sufficient for e
    xploratory analysis or quick comparisons, while finer sampling is useful
    when trying to match specific experimental views or when preparing
    publication-quality figures.

    ## Symmetry Considerations

    The symmetry group defines how orientations are reduced according to the
    intrinsic symmetry of the structure.

    For asymmetric particles, the default **c1** symmetry should be used. For
    symmetric assemblies (for example cyclic or dihedral symmetries), specifying
    the correct symmetry ensures that equivalent orientations are not
    redundantly sampled.

    Biologically, this is important because symmetry determines which views are
    unique. Using the wrong symmetry may lead to misleading interpretations
    when comparing projections to experimental data.

    ## Projection Methods

    The protocol offers several methods to compute projections, which mainly
    differ in computational approach and numerical properties.

    The **Fourier method** is the default and most commonly used option. It
    computes projections using the central slice theorem in Fourier space and
    provides a good balance between accuracy and efficiency. For most
    biological applications, this is the recommended choice.

    The **Real space method** performs projections by integrating along rays
    through the volume. It is conceptually straightforward but generally slower.

    The **Shears method** is an alternative real-space approach that can be
    efficient in certain situations but is less commonly used in routine
     workflows.

    In most cases, users do not need to change the default method unless there
    is a specific reason to explore numerical differences.

    ## Advanced Parameters

    Several advanced parameters are available, mainly affecting the
    Fourier-based projection.

    The **padding factor** controls how much the volume is expanded before
    projection. Increasing padding can improve interpolation accuracy but also
    increases computational cost.

    The **maximum frequency** defines the highest spatial frequency considered
    in the projection. Limiting this value can reduce noise or numerical
    artifacts but may also remove high-resolution information.

    The **interpolation method** determines how values are estimated between
    sampled points. Higher-order interpolation (such as cubic B-spline)
    typically produces smoother and more accurate projections.

    Another parameter, **shift sigma**, introduces small random shifts in the
    projections. While not commonly used in standard workflows, it can be
    helpful when simulating more realistic image variability.

    For most biological use cases, the default values of these parameters
    are appropriate.

    ## Outputs and Their Interpretation

    The protocol produces a set of 2D images corresponding to projections of
    the input volume. Each image is associated with a specific orientation,
    and together they form a gallery covering the sampled angular space.

    These projections can be directly compared with experimental particle
    images or 2D class averages. Good agreement between projections and
    experimental data supports the validity of the reconstructed volume.

    From a biological perspective, the gallery helps to:
    * Identify characteristic views of the structure
    * Understand how structural features appear in projection
    * Detect missing or underrepresented orientations in the data

    ## Practical Recommendations

    A typical use of this protocol is to generate projections after obtaining a
    3D reconstruction and compare them visually with 2D class averages. This
    helps verify that the reconstruction explains the observed data.

    For exploratory analysis, moderate angular steps (for example 5–10 degrees)
    are usually sufficient. For more detailed comparisons or figure
    preparation, smaller steps may be beneficial.

    If the structure has known symmetry, it is important to specify it
    correctly to avoid redundant projections and to obtain a biologically
    meaningful gallery.

    In most cases, the default Fourier projection method with standard
    parameters provides reliable results without further tuning.

    ## Final Perspective

    The Create Gallery protocol is a simple but powerful tool for connecting 3D
    structures with their 2D experimental observations. By visualizing how a
    volume appears from different orientations, it provides valuable intuition
    and serves as a practical validation step in cryo-EM workflows.

    For biological users, it is often one of the most direct ways to assess
    whether a reconstruction truly reflects the underlying data.
    """
    _devStatus = PROD

    _label = 'create gallery'
    _version = VERSION_1_1

    PARAM_FILE_NAME =  "projectionParameters.xmd"
    METHOD_REAL_SPACE = 0
    METHOD_SHEARS= 1
    METHOD_FOURIER = 2
    INTERP_METHOD_NEAREST = 0
    INTERP_METHOD_LINEAR = 1
    INTERP_METHOD_BSPLINE = 2

    interpMethodsDict = {
        INTERP_METHOD_NEAREST: "nearest",
        INTERP_METHOD_LINEAR: "linear",
        INTERP_METHOD_BSPLINE: "bspline"
        }

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')

        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label='Input volume')
        
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See'
                           'https://github.com/I2PC/xmipp-portal/wiki/Symmetry '
                           'for a description of the symmetry groups format. '
                           'If no symmetry is present, give c1')

        rot = form.addLine('Rotational angle',
                           help='Minimum, maximum and step values for '
                                'rotational angle range, all in degrees.')
        rot.addParam('rot0', FloatParam, default=0, label='Min')
        rot.addParam('rotF', FloatParam, default=360, label='Max')
        rot.addParam('rotStep', FloatParam, default=5, label='Step')

        tilt = form.addLine('Tilt angle',
                            help='In degrees. tilt=0 is a top view, '
                                 'while tilt=90 is a side view"')
        tilt.addParam('tilt0', FloatParam, default=0, label='Min')
        tilt.addParam('tiltF', FloatParam, default=180, label='Max')
        tilt.addParam('tiltStep', FloatParam, default=5, label='Step')

        form.addParam('shiftSigma',FloatParam, default=0.0,
                      expertLevel=LEVEL_ADVANCED,
                      label='Shift sigma', help="In pixels")

        form.addParam('projectionMethod',
                EnumParam,
                choices=['Real space', 'Shears', 'Fourier'],
                default=2,
                label='Projection method',
                help='Method used for computing the projection:\n'
                    '- Real space: Makes projections by ray tracing in real space.\n'
                    '- Shears: Use real-shears algorithm.\n'
                    '- Fourier: Takes a central slice in Fourier space.')

        form.addParam('pad',FloatParam, default=2,
                      expertLevel=LEVEL_ADVANCED,
                      label='Padding', 
                      help='When calculating the proyection with the Fourier method, '
                           'it controls the padding factor.',
                      condition='projectionMethod==2')
        
        form.addParam('maxFreq',FloatParam, default=0.25,
                      expertLevel=LEVEL_ADVANCED,
                      label='Maximum frequency', 
                      help='When calculating the proyection with the Fourier method, '
                           'it is the maximum frequency for the pixels and by default '
                           'pixels with frequency more than 0.25 are not considered.',
                      condition='projectionMethod==2')
        
        form.addParam('interpolationMethod',
                      EnumParam,
                      label='Interpolation method',
                      choices=['Nearest Neighborhood', 'Linear BSpline', 'Cubic BSpline'],
                      default=2,
                      help='When calculating the proyection with the Fourier method, '
                           'it is the method for interpolation.',
                      condition='projectionMethod==2')

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('copyInput')
        self._insertFunctionStep('createGallery')
        self._insertFunctionStep('createOutput')
    
    #--------------------------- STEPS functions -------------------------------
    def copyInput(self):
        ImageHandler().convert(self.inputVolume.get(),
                               self._getTmpPath("volume.vol"))
                        
    def createGallery(self):
        xdim = self.inputVolume.get().getXDim()
        rotN = round((self.rotF.get()-self.rot0.get())/self.rotStep.get())
        tiltN = round((self.tiltF.get()-self.tilt0.get())/self.tiltStep.get())

        paramContent ="""# XMIPP_STAR_1 *
data_block1
_dimensions2D   '%d %d'
_projRotRange    '%f %f %d'
_projRotRandomness   even 
_projTiltRange    '%f %f %d'
_projTiltRandomness   even 
_projPsiRange    '0 0 1'
_projPsiRandomness   even 
_noiseCoord '%f 0'
""" % (xdim, xdim, self.rot0, self.rotF,rotN, self.tilt0, self.tiltF, tiltN, self.shiftSigma)
        
        fhParam = open(self._getExtraPath(self.PARAM_FILE_NAME), 'w')
        fhParam.write(paramContent)
        fhParam.close()

        params = {
            'i': self._getTmpPath("volume.vol"),
            'o': self._getPath("images.stk"),
            'params': self._getExtraPath(self.PARAM_FILE_NAME),
            'sym': self.symmetryGroup,
        }

        args = "-i %(i)s " \
               "-o %(o)s " \
               "--params %(params)s " \
               "--sym %(sym)s "
        
        if self.projectionMethod.get() == self.METHOD_REAL_SPACE:
            params['method'] = "real_space"
            args += "--method %(method)s "
        elif self.projectionMethod.get() == self.METHOD_SHEARS:
            params['method'] = "shears"
            args += "--method %(method)s "
        elif self.projectionMethod.get() == self.METHOD_FOURIER:
            params['method'] = "fourier"
            params['pad'] = self.pad
            params['maxFreq'] = self.maxFreq
            params['interp'] = self.interpMethodsDict[self.interpolationMethod.get()]
            args += "--method %(method)s %(pad)f %(maxFreq)f %(interp)s" \

        self.runJob("xmipp_phantom_project", args % params)

    def createOutput(self):
        imgSetOut = self._createSetOfAverages()
        imgSetOut.setSamplingRate(self.inputVolume.get().getSamplingRate())
        imgSetOut.setAlignmentProj()
        readSetOfParticles(self._getPath("images.xmd"), imgSetOut)

        self._defineOutputs(outputReprojections=imgSetOut)
        self._defineSourceRelation(self.inputVolume, imgSetOut)

    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        messages = []
        messages.append("Rot.angle from %0.2f to %0.2f in steps of %0.2f" %
                        (self.rot0, self.rotF, self.rotStep))
        messages.append("Tilt.angle from %0.2f to %0.2f in steps of %0.2f" %
                        (self.tilt0, self.tiltF, self.tiltStep))
        return messages
