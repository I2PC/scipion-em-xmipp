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
