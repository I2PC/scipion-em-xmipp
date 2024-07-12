# **************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
# *              Javier Mota (jmota@cnb.csic.es)
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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, BooleanParam
from pyworkflow.utils.path import cleanPath
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Image
import pwem.emlib.metadata as md

from pwem import emlib
from xmipp3.convert import setXmippAttributes, xmippToLocation

        
class XmippProtGenerateReprojections(ProtAnalysis3D):
    """Compares a set of classes or averages with the corresponding projections of a reference volume.
    The set of images must have a 3D angular assignment and the protocol computes the residues
    (the difference between the experimental images and the reprojections). The zscore of the mean
    and variance of the residues are computed. Large values of these scores may indicate outliers.
    The protocol also analyze the covariance matrix of the residual and computes the logarithm of
    its determinant [Cherian2013]. The extremes of this score (called zScoreResCov), that is
    values particularly low or high, may indicate outliers."""
    _label = 'generate reprojections'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume to compare images to", important=True,
                      pointerClass='Volume',
                      help='Volume to be used for class comparison')
        form.addParam('ignoreCTF', BooleanParam, default=True, label='Ignore CTF',
                      help='By ignoring the CTF you will create projections more similar to what a person expects, '
                           'while by using the CTF you will create projections more similar to what the microscope sees')
        form.addParallelSection(threads=0, mpi=8)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        vol = self.inputVolume.get()
        
        self._insertFunctionStep("convertStep")
        imgSet = self.inputSet.get()
        anglesFn = self.imgsFn
        self._insertFunctionStep("produceProjections",
                                 anglesFn,
                                 vol.getSamplingRate())
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from xmipp3.convert import writeSetOfParticles
        imgSet = self.inputSet.get()
        writeSetOfParticles(imgSet, self.imgsFn)

        from pwem.emlib.image import ImageHandler
        img = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xdim = self.inputVolume.get().getDim()[0]

        imgXdim = imgSet.getDim()[0]
        if xdim != imgXdim:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" % (fnVol, imgXdim), numberOfMpi=1)
    
    def produceProjections(self, fnAngles, Ts):
        fnVol = self._getTmpPath("volume.vol")
        anglesOutFn = self._getExtraPath("anglesCont.stk")
        projectionsOutFn = self._getExtraPath("projections.stk")
        args = "-i %s -o %s --ref %s --oprojections %s --sampling %f " %\
             (fnAngles, anglesOutFn, fnVol, projectionsOutFn, Ts)
        if self.ignoreCTF:
            args += " --ignoreCTF "
        self.runJob("xmipp_angular_continuous_assign2", args)
        fnNewParticles = self._getExtraPath("images.stk")
        if os.path.exists(fnNewParticles):
            cleanPath(fnNewParticles)
    
    def createOutputStep(self):

        fnImgs = self._getExtraPath('images.stk')
        if os.path.exists(fnImgs):
            cleanPath(fnImgs)

        imgSet = self.inputSet.get()
        imgFn = self._getExtraPath("anglesCont.xmd")

        self.newAssignmentPerformed = os.path.exists(self._getExtraPath("angles.xmd"))
        self.samplingRate = self.inputSet.get().getSamplingRate()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        if not self.newAssignmentPerformed:
            outputSet.setAlignmentProj()
        self.iterMd = md.iterRows(imgFn, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(imgSet,
                            updateItemCallback=self._updateItem
                            )
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

        imgSet = self.inputSet.get()
        outputSet2 = self._createSetOfParticles('2')
        outputSet2.copyInfo(imgSet)
        if not self.newAssignmentPerformed:
            outputSet2.setAlignmentProj()
        self.iterMd2 = md.iterRows(imgFn, md.MDL_ITEM_ID)
        self.lastRow2 = next(self.iterMd2)
        outputSet2.copyItems(imgSet,
                            updateItemCallback=self._updateItem2,
                            )
        self._defineOutputs(outputProjections=outputSet2)
        self._defineSourceRelation(self.inputSet, outputSet2)

    def _updateItem(self, particle, row):
        count = 0

        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                md.MDL_ITEM_ID):
            count += 1
            if count:
                self._processRow(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0

    def _updateItem2(self, particle2, row):
        count = 0

        while self.lastRow2 and particle2.getObjId() == self.lastRow2.getValue(
                md.MDL_ITEM_ID):
            count += 1
            if count:
                self._processRow2(particle2, self.lastRow2)
            try:
                self.lastRow2 = next(self.iterMd2)
            except StopIteration:
                self.lastRow2 = None

        particle2._appendItem = count > 0

    def _processRow(self, particle, row):
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))

        particle.setLocation(xmippToLocation(row.getValue(emlib.MDL_IMAGE)))
        #__setXmippImage(emlib.MDL_IMAGE)
        #__setXmippImage(emlib.MDL_IMAGE_REF)

        setXmippAttributes(particle, row, emlib.MDL_IMAGE_ORIGINAL,
                           emlib.MDL_IMAGE_REF)

    def _processRow2(self, particle, row):
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))

        particle.setLocation(xmippToLocation(row.getValue(
            emlib.MDL_IMAGE_REF)))
        #__setXmippImage(emlib.MDL_IMAGE)
        #__setXmippImage(emlib.MDL_IMAGE_REF)

        setXmippAttributes(particle, row, emlib.MDL_IMAGE_ORIGINAL,
                           emlib.MDL_IMAGE_REF)


    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append(
                "We projected the volume %s following the directions in %i input images %s." \
                % (
                self.getObjectTag('inputVolume'), self.inputSet.get().getSize(),
                self.getObjectTag('inputSet')))
        return methods

