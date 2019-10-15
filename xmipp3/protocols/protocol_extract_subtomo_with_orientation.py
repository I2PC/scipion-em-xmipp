# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez
# *              Carlos Oscar Sanchez Sorzano
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

from pyworkflow.em.convert import ImageHandler
from pyworkflow.em.data import SetOfVolumes
import pyworkflow.em.metadata as md
from pyworkflow.em.protocol import EMProtocol
from pyworkflow.protocol.params import PointerParam, BooleanParam, FloatParam
from tomo.objects import SetOfSubTomograms
from xmipp3.convert import alignmentToRow
import xmippLib


class XmippProtExtractSubtomoOrient(EMProtocol):
    """ This protocol extracts subtomograms with orientation from a tomogram."""

    _label = 'extract oriented subtomos'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputTomogram', PointerParam, pointerClass="Tomogram", label='Tomogram',
                      help="Tomogram from which the subtomograms will be extracted.")
        form.addParam('inputCoordinates', PointerParam, pointerClass="SetOfCoordinates3D", label='Coordinates',
                      help="Select the SetOfCoordinates3D. ")
        form.addParam('boxSize', FloatParam, label='Box size',
                      help='The subtomograms are extracted as a cubic box of this size. ')
        form.addParam('invertContrast', BooleanParam, default=False, label='Invert contrast of extracted subtomograms',
                      help= "Invert the contrast if the reference is black over a white background.  Xmipp, Spider, "
                            "Relion and Eman require white particles over a black background. ")
        form.addParam('downFactor', FloatParam, default=1.0, label='Downsampling factor',
                      help='Select a value greater than 1.0 to reduce the size of subtomograms after extraction. '
                           'If 1.0 is used, no downsample is applied. Non-integer downsample factors are possible. ')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInput', self.inputTomogram.getObjId())
        self._insertFunctionStep('runExtract')
        self._insertFunctionStep('postProcessing')
        self._insertFunctionStep('createOutput')

    # --------------------------- STEPS functions -------------------------------
    def convertInput(self, objIdTomo):
        img = ImageHandler()
        fnTomo = self._getExtraPath('tomogram.mrc')
        img.convert(self.inputTomogram.get(), fnTomo)

    def runExtract(self):
        TsCoors = self.inputCoordinates.get().getSamplingRate()
        TsTomo = self.inputTomogram.get().getSamplingRate()
        scaleFactor = TsCoors/TsTomo
        mdGeometry = xmippLib.MetaData()
        for coor in self.inputCoordinates.get().iterItems():
            nRow = md.Row()
            nRow.setValue(xmippLib.MDL_ITEM_ID,long(coor.getObjId()))
            nRow.setValue(xmippLib.MDL_XCOOR,int(coor.getX()*scaleFactor))
            nRow.setValue(xmippLib.MDL_YCOOR,int(coor.getY()*scaleFactor))
            nRow.setValue(xmippLib.MDL_ZCOOR,int(coor.getZ()*scaleFactor))
            # Convert transform matrix to Euler Angles (rot, tilt, psi) ==> INVERT!
            # alignmentToRow(coor.getTransform(),nRow,ALIGN_3D)
            nRow.addToMd(mdGeometry)
        fnGeometry = self._getExtraPath("geometry%d.xmd")
        mdGeometry.write(fnGeometry)
        args = " -i %s -o %s --geom %s --box %d" % (self._getExtraPath("tomogram.mrc"),
                                                    self._getExtraPath("subtomograms.mrc"),
                                                    self._getExtraPath("geometry.xmd"), self.boxSize.get())
        # self.runJob("xmipp_tomo_extract_orient",args)

    def postProcessing(self):
        subtomograms = self._getExtraPath("subtomograms.mrc")
        if self.invertContrast.get() == True:
            self.runJob("xmipp_image_operate", " -i %s  --mult -1" % subtomograms)
        if self.downFactor.get() != 1:
            I = xmippLib.Image(subtomograms)
            x, y, z, _ = I.getDimensions()
            I.scale(int(x / self.downFactor.get()), int(y / self.downFactor.get()), int(z / self.downFactor.get()))
            I.write(self._getExtraPath("downsampled_subtomograms.mrc"))

    def createOutput(self):
        outputSubTomogramsSet = self.createSetOfSubTomograms()
        outputSubTomogramsSet.setSamplingRate(self.inputTomogram.get().getSamplingRate())
        # outputSubTomogramsSet.setCoordinates3D(self.inputCoordinates) # EACH SUBTOMO HAS TO HAVE ITS COORDINATES
        # outputSubTomogramsSet.setLocation(self._getExtraPath("subtomograms.mrc")) # EACH SUBTOMO HAS TO HAVE ITS LOCATION
        self._defineOutputs(outputSubtomograms=outputSubTomogramsSet)
        self._defineSourceRelation(self.inputTomogram, outputSubTomogramsSet)
        self._defineSourceRelation(self.inputCoordinates, outputSubTomogramsSet)

    # --------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        summary.append("%d subtomograms extracted with box size %d" %
                       (len(self.inputCoordinates.get()),self.boxSize.get()))
        return summary

    def _methods(self):
        methods = []
        methods.append("%d subtomograms extracted with orientation with a box size of %d from tomogram %s " %
                       (len(self.inputCoordinates.get()),self.boxSize.get(), self.inputTomogram.get().getFileName()))
        return methods