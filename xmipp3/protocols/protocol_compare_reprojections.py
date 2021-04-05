# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
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

from math import floor
import os

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam)
from pyworkflow.utils.path import cleanPath
from pwem.protocols import ProtAnalysis3D
from pwem.objects import (SetOfClasses2D, Image, SetOfAverages, SetOfParticles)
import pwem.emlib.metadata as md
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem import emlib
from xmipp3.base import ProjMatcher
from xmipp3.convert import setXmippAttributes, xmippToLocation

        
class XmippProtCompareReprojections(ProtAnalysis3D, ProjMatcher):
    """Compares a set of classes or averages with the corresponding projections of a reference volume.
    The set of images must have a 3D angular assignment and the protocol computes the residues
    (the difference between the experimental images and the reprojections). The zscore of the mean
    and variance of the residues are computed. Large values of these scores may indicate outliers.
    The protocol also analyze the covariance matrix of the residual and computes the logarithm of
    its determinant [Cherian2013]. The extremes of this score (called zScoreResCov), that is
    values particularly low or high, may indicate outliers."""
    _label = 'compare reprojections'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfClasses2D, SetOfAverages, SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume to compare images to", important=True,
                      pointerClass='Volume',
                      help='Volume to be used for class comparison')
        form.addParam('useAssignment', BooleanParam, default=True,
                      label='Use input angular assignment (if available)')
        form.addParam('optimizeGray', BooleanParam, default=False,
                      label='Optimize gray scale')
        form.addParam('ignoreCTF', BooleanParam, default=True,
                      label='Ignore CTF',
                      help='By ignoring the CTF you will create projections more similar to what a person expects, '
                           'while by using the CTF you will create projections more similar to what the microscope sees')
        form.addParam('evaluateResiduals', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Evaluate residuals',
                      help='If this option is chosen, then the residual covariance matrix is calculated and characterized. '
                           'But this option takes time and disk space')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
        form.addParam('angularSampling', FloatParam, default=5, expertLevel=LEVEL_ADVANCED,
                      label='Angular sampling rate',
                      help='In degrees.'
                      ' This sampling defines how fine the projection gallery from the volume is explored.')
        form.addParallelSection(threads=0, mpi=8)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        vol = self.inputVolume.get()
        
        self._insertFunctionStep("convertStep", self.imgsFn)
        imgSet = self.inputSet.get()
        if not self.useAssignment or isinstance(imgSet, SetOfClasses2D) or isinstance(imgSet, SetOfAverages) or (isinstance(imgSet, SetOfParticles) and not imgSet.hasAlignmentProj()):
            anglesFn = self._getExtraPath('angles.xmd')
            self._insertFunctionStep("projMatchStep", self.inputVolume.get().getFileName(), self.angularSampling.get(), self.symmetryGroup.get(), self.imgsFn,
                                     anglesFn, self.inputVolume.get().getDim()[0])
        else:
            anglesFn=self.imgsFn

        self._insertFunctionStep("produceResiduals", vol.getFileName(), anglesFn, vol.getSamplingRate())
        if self.evaluateResiduals.get():
            self._insertFunctionStep("evaluateResiduals")
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, imgsFn):
        from ..convert import writeSetOfClasses2D, writeSetOfParticles
        imgSet = self.inputSet.get()
        if isinstance(imgSet, SetOfClasses2D):
            writeSetOfClasses2D(imgSet, self.imgsFn, writeParticles=True)
        else:
            writeSetOfParticles(imgSet, self.imgsFn)
        from pwem.emlib.image import ImageHandler
        img = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xdim=self.inputVolume.get().getDim()[0]
        if xdim!=self._getDimensions():
            self.runJob("xmipp_image_resize","-i %s --dim %d"%(fnVol,self._getDimensions()),numberOfMpi=1)
    
    def produceResiduals(self, fnVol, fnAngles, Ts):
        fnVol = self._getTmpPath("volume.vol")
        anglesOutFn=self._getExtraPath("anglesCont.stk")

        projectionsOutFn=self._getExtraPath("projections.stk")
        xdim=self._getDimensions()
        args="-i %s -o %s --ref %s --optimizeAngles --optimizeShift --max_shift %d --oprojections %s --sampling %f"%\
                    (fnAngles,anglesOutFn,fnVol,floor(xdim*0.05),projectionsOutFn,Ts)
        if self.evaluateResiduals:
            args+=" --oresiduals %s"%self._getExtraPath("residuals.stk")
        if self.ignoreCTF:
            args+=" --ignoreCTF"
        if self.optimizeGray:
            args+="--optimizeGray --max_gray_scale 0.95 "
        self.runJob("xmipp_angular_continuous_assign2", args)
        fnNewParticles=self._getExtraPath("images.stk")
        if os.path.exists(fnNewParticles):
            cleanPath(fnNewParticles)
    
    def evaluateResiduals(self):
        # Evaluate each image
        fnAutoCorrelations = self._getExtraPath("autocorrelations.xmd")
        stkAutoCorrelations = self._getExtraPath("autocorrelations.stk")
        stkResiduals = self._getExtraPath("residuals.stk")
        anglesOutFn=self._getExtraPath("anglesCont.xmd")
        self.runJob("xmipp_image_residuals", " -i %s -o %s --save_metadata_stack %s" % (stkResiduals, stkAutoCorrelations, fnAutoCorrelations), numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities", '-i %s --operate rename_column "image imageResidual"' % fnAutoCorrelations, numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities", '-i %s --set join %s imageResidual' % (anglesOutFn, fnAutoCorrelations), numberOfMpi=1)
        cleanPath(fnAutoCorrelations)
    
    def createOutputStep(self):
        fnImgs = self._getExtraPath('images.stk')
        if os.path.exists(fnImgs):
            cleanPath(fnImgs)

        imgSet = self.inputSet.get()
        imgFn = self._getExtraPath("anglesCont.xmd")
        self.newAssignmentPerformed = os.path.exists(self._getExtraPath("angles.xmd"))
        self.samplingRate = self.inputSet.get().getSamplingRate()

        # Special case for 2D classes
        if isinstance(imgSet, SetOfClasses2D):
            outputSet = self._createSetOfClasses2D(imgSet.getImages())
            outputSet.copyInfo(imgSet)
            outputSet.appendFromClasses(imgSet, updateClassCallback=lambda clazz: self._updateClass(clazz, imgFn))

        # Particles or Averages
        else:
            if isinstance(imgSet, SetOfAverages):
                outputSet = self._createSetOfAverages()
                outputSet.copyInfo(imgSet)
            else:
                outputSet = self._createSetOfParticles()
                outputSet.copyInfo(imgSet)
                if not self.newAssignmentPerformed:
                    outputSet.setAlignmentProj()
            self.iterMd = md.iterRows(imgFn, md.MDL_ITEM_ID)
            self.lastRow = next(self.iterMd)
            outputSet.copyItems(imgSet,
                                updateItemCallback=self._processRow)

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

    def _updateClass(self, clazz, mdFile):
        """ Callback to update the class"""

        classId = clazz.getObjId()

        if classId not in self._classesInfo:
            self._classesInfo[classId] = clazz
            # Get the row
            row = self._getMdRow(mdFile, classId)
            self._createItemMatrix(clazz, row)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise Exception("Missing row %s at %s" % (id,mdFile))
    def _processRow(self, particle, row):
        count = 0
        
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
                    
        particle._appendItem = count > 0

    def _createItemMatrix(self, particle, row):
        setXmippAttributes(particle, row,
                           emlib.MDL_COST, emlib.MDL_CONTINUOUS_GRAY_A,
                           emlib.MDL_CONTINUOUS_GRAY_B, emlib.MDL_CONTINUOUS_X,
                           emlib.MDL_CONTINUOUS_Y,
                           emlib.MDL_CORRELATION_IDX, emlib.MDL_CORRELATION_MASK,
                           emlib.MDL_CORRELATION_WEIGHT, emlib.MDL_IMED)
        if self.evaluateResiduals:
            setXmippAttributes(particle, row,
                               emlib.MDL_ZSCORE_RESVAR, emlib.MDL_ZSCORE_RESMEAN,
                               emlib.MDL_ZSCORE_RESCOV)
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))
        
        __setXmippImage(emlib.MDL_IMAGE)
        __setXmippImage(emlib.MDL_IMAGE_REF)
        if self.evaluateResiduals:
            __setXmippImage(emlib.MDL_IMAGE_RESIDUAL)
            __setXmippImage(emlib.MDL_IMAGE_COVARIANCE)

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We evaluated %i input images %s regarding to volume %s."\
                           %(self.inputSet.get().getSize(), self.getObjectTag('inputSet'), self.getObjectTag('inputVolume')) )
            methods.append("The residuals were evaluated according to their mean, variance and covariance structure [Cherian2013].")
        return methods
    
    #--------------------------- UTILS functions --------------------------------------------
    def _getDimensions(self):
        imgSet = self.inputSet.get()
        if isinstance(imgSet, SetOfClasses2D):
            xDim = imgSet.getImages().getDim()[0]
        else:
            xDim = imgSet.getDim()[0]
        return xDim
