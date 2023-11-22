# coding=utf-8
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *              Carlos Oscar Sanchez Sorzano
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, FloatParam, BooleanParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import cleanPattern
from pwem.protocols import ProtAnalysis3D
from pwem import emlib

from xmipp3.convert import readSetOfParticles, writeSetOfParticles

CITE ='Fernandez-Gimenez2023b'


class XmippProtLocalCTF(ProtAnalysis3D):
    """Compares a set of particles with the corresponding projections of a reference volume.
    The set of particles must have a 3D angular assignment.
    This protocol refines the CTF, computing local defocus change.
    The maximun allowed defocus is a parameter introduced by the user (advanced).
    The protocol gives back the input set of particles with the refine local defocus and the defocus change with relation to the global defocus."""
    _label = 'estimate local defocus'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj')
        form.addParam('inputVolume', PointerParam, label="Volume to compare images to",
                      pointerClass='Volume',
                      help='Volume to be used for class comparison')
        form.addParam('maxDefocusChange', FloatParam, label="Maximum defocus change (A)", default=500,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('maxGrayScaleChange', FloatParam, label="Maximum gray scale change", default=1,
                      expertLevel=LEVEL_ADVANCED, help="The reprojection is modified as a*P+b, a is restricted to the "
                                                       "interval [1-maxGrayScale,1+maxGrayScale]")
        form.addParam('maxGrayShiftChange', FloatParam, label="Maximum gray shift change", default=1,
                      expertLevel=LEVEL_ADVANCED, help="The reprojection is modified as a*P+b, b is restricted to the "
                                                       "interval [-maxGrayShift,maxGrayShift]")
        form.addParam('sameDefocus', BooleanParam, label="Force defocusV to be equal than defocusU", default=False,
                      expertLevel=LEVEL_ADVANCED,
                      help="As the CTF usually suffers from astigmatism (it is not spherical but ellipsoidal), the "
                           "defocus vary if computed in X or Y direction, being defocus U value the defocus in X "
                           "direction and defocus V value the defocus in Y direction.")
        form.addParallelSection(threads=0, mpi=8)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("refineDefocus")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        """convert input to proper format and dimensions if necessary"""
        imgSet = self.inputSet.get()
        writeSetOfParticles(imgSet, self._getExtraPath('input_imgs.xmd'))
        img = emlib.image.ImageHandler()
        fnVol = self._getExtraPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xDimVol=self.inputVolume.get().getDim()[0]
        xDimImg = imgSet.getDim()[0]
        if xDimVol!=xDimImg:
            self.runJob("xmipp_image_resize", "-i %s --dim %d"%(fnVol,xDimImg),numberOfMpi=1)

    def refineDefocus(self):
        """compute local defocus using Xmipp (xmipp_angular_continuous_assign2) and add to metadata columns related to
        defocus"""
        fnVol = self._getExtraPath("volume.vol")
        fnIn = self._getExtraPath('input_imgs.xmd')
        fnOut = self._getExtraPath('output_imgs.xmd')
        anglesOutFn = self._getExtraPath("anglesCont.stk")
        Ts = self.inputSet.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --optimizeDefocus --max_defocus_change %d --sampling %f --optimizeGray " \
               "--max_gray_scale %f --max_gray_shift %f " % \
               (fnIn, anglesOutFn, fnVol, self.maxDefocusChange.get(), Ts, self.maxGrayScaleChange.get(),
                self.maxGrayShiftChange.get())
        if self.inputSet.get().isPhaseFlipped():
            args += " --phaseFlipped"
        if self.sameDefocus.get():
            args += " --sameDefocus"
        self.runJob("xmipp_angular_continuous_assign2", args)

        fnCont = self._getExtraPath('anglesCont.xmd')
        self.runJob("xmipp_metadata_utilities", '-i %s --operate keep_column "itemId ctfDefocusU ctfDefocusV '
                                                'ctfDefocusChange ctfDefocusAngle"'%
                    fnCont, numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    '-i %s -o %s --operate drop_column "ctfDefocusU ctfDefocusV ctfDefocusChange ctfDefocusAngle"' %
                    (fnIn, fnOut), numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --set join %s itemId itemId" % (fnOut, fnCont), numberOfMpi=1)

        cleanPattern(self._getExtraPath("anglesCont.*"))

    def createOutputStep(self):
        """create scipion output data from metadata"""
        outputSet = self._createSetOfParticles()
        imgSet = self.inputSet.get()
        outputSet.copyInfo(imgSet)
        readSetOfParticles(self._getExtraPath('output_imgs.xmd'), outputSet,
                           extraLabels=[emlib.MDL_CTF_DEFOCUS_CHANGE])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Refined defocus of %i particles" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        summary.append("Allowed defocus: %s" % self.maxDefocusChange.get())
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We refined the defocus %i input particles %s regarding to volume %s, allowing a maximun "
                           "defocus of %s" % (self.inputSet.get().getSize(), self.getObjectTag('inputSet'),
                                              self.getObjectTag('inputVolume'), self.maxDefocusChange.get()))
        return methods
