# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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

from pwem.emlib.image import ImageHandler as ih
from pwem.objects import Volume
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, EnumParam, FloatParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
import pyworkflow.object as pwobj
from pwem import ALIGN_PROJ
import pwem.emlib.metadata as md
from pwem.protocols import EMProtocol
from xmipp3.convert import xmippToLocation, writeSetOfParticles, readSetOfParticles


class XmippProtShiftParticles(EMProtocol):
    """ This protocol shifts particles to center them into a point selected in a volume. To do so, it generates new
    shifted images and modify the transformation matrix according to the shift performed."""

    _label = 'shift particles'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles', label="Particles",
                      help='Select the SetOfParticles with transformation matrix to be shifted.')
        form.addParam('option', BooleanParam, label='Select position in volume?', default='True',
                      help='Select the position where the particles will be shifted in a volume displayed in a wizard.')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", allowsNull=True,
                      condition='option', help='Volume to select the point (by clicking in the wizard for selecting the'
                                               ' new center) that will be the new center of the particles.')
        form.addParam('xin', FloatParam, label="x", condition='option',
                      help='Use the wizard to select the new center for the shifted particles by shift+click on the '
                           'blue point a drag it to the desired location while pressing shift.')
        form.addParam('yin', FloatParam, label="y", condition='option')
        form.addParam('zin', FloatParam, label="z", condition='option')
        form.addParam('inputMask', PointerParam, pointerClass='VolumeMask', label="Volume mask", allowsNull=True,
                      condition='not option', help='3D mask to compute the center of mass, the particles will be '
                                                   'shifted to the computed center of mass')
        form.addParam('applyShift', BooleanParam, label='Apply shift to particles?', default='True',
                      help='Yes: The shift is applied to particle images and zero shift is stored in the metadata. No: '
                           'The shift is stored in the transformation matrix in the metadata, but not applied to the '
                           'particle image (i.e. the output images are the same of input images). This option takes '
                           'less time and the shift could be applied later using protocol "xmipp3 - apply alignment 2d"'
                           ' or by re-extracting the particles.')
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted particles?',
                      default='True', help='Use input particles box size for the shifted particles.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size for the shifted particles.')
        form.addParam('inv', BooleanParam, label='Inverse', expertLevel=LEVEL_ADVANCED, default='True',
                      help='Use inverse transformation matrix')
        form.addParam('interp', EnumParam, default=0, choices=['Linear', 'Spline'], expertLevel=LEVEL_ADVANCED,
                      display=EnumParam.DISPLAY_HLIST, label='Interpolation',
                      help='Linear: Use bilinear/trilinear interpolation\nSpline: Use spline interpolation')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        """convert input particles into .xmd file """
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath("input_particles.xmd"))

    def shiftStep(self):
        """call xmipp program to shift the particles"""
        centermd = self._getExtraPath("center_particles.xmd")
        args = '-i "%s" -o "%s" ' % (self._getExtraPath("input_particles.xmd"), centermd)
        if self.option:
            self.x = self.xin.get()
            self.y = self.yin.get()
            self.z = self.zin.get()
        else:
            fnvol = self.inputMask.get().getFileName()
            if fnvol.endswith('.mrc'):
                fnvol += ':mrc'
            vol = Volume()
            vol.setFileName(fnvol)
            vol = ih().read(vol.getFileName())
            masscenter = vol.centerOfMass()
            self.x = masscenter[0] - 0.5 * vol.getDimensions()[0]
            self.y = masscenter[1] - 0.5 * vol.getDimensions()[1]
            self.z = masscenter[2] - 0.5 * vol.getDimensions()[2]

        args += '--shift_to %f %f %f ' % (self.x, self.y, self.z)
        program = "xmipp_transform_geometry"
        if not self.interp.get():
            interp = 'linear'
        else:
            interp = 'spline'
        if self.applyShift.get():
            args += ' --apply_transform'
        args += ' --dont_wrap --interp %s' % interp
        if self.inv.get():
            args += ' --inverse'
        self.runJob(program, args)

        if not self.boxSizeBool.get():
            box = self.boxSize.get()
            self.runJob('xmipp_transform_window', '-i "%s" -o "%s" --size %d %d %d --save_metadata_stack' %
                        (centermd, self._getExtraPath("crop_particles.stk"), box, box, 1))

    def createOutputStep(self):
        """create output with the new particles"""
        self.ix = 0
        inputParticles = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputParticles)
        if self.boxSizeBool.get():
            outputmd = self._getExtraPath("center_particles.xmd")
        else:
            outputmd = self._getExtraPath("crop_particles.xmd")
        readSetOfParticles(outputmd, outputSet, alignType=ALIGN_PROJ)
        self._defineOutputs(outputParticles=outputSet)
        self._defineOutputs(shiftX=pwobj.Float(self.x),
                            shiftY=pwobj.Float(self.y),
                            shiftZ=pwobj.Float(self.z))
        self._defineSourceRelation(inputParticles, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            if self.option:
                option = "User defined shift"
            else:
                option = "Shift to center of mass"
            if not self.interp.get():
                interp = 'linear'
            else:
                interp = 'spline'
            summary.append("%s\ninterpolation: %s" % (option, interp))
            if self.inv.get():
                summary.append("inverse matrix applied")
        return summary

    def _validate(self):
        for part in self.inputParticles.get().iterItems():
            if not part.hasTransform():
                validatemsg = ['Please provide particles which have transformation matrix.']
                return validatemsg
