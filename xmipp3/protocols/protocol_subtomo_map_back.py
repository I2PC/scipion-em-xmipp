# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Carlos Oscar Sanchez Sorzano
# *              Estrella Fernandez Gimenez
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

import numpy as np

from pyworkflow.em.protocol import EMProtocol
from pyworkflow.protocol.params import PointerParam, EnumParam, BooleanParam, FloatParam
from pyworkflow.em.convert import ImageHandler
import xmippLib

class XmippProtSubtomoMapBack(EMProtocol):
    """ This protocol takes a tomogram, a reference subtomogram and a metadata with geometrical parameters
   (x,y,z) and places the reference subtomogram on the tomogram at the designated locations (map back) """

    _label = 'subtomogram map back'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)


    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input subtomograms')
        form.addParam('inputSubtomograms', PointerParam, pointerClass="SetOfSubTomograms",
                      label='Set of subtomograms', help="Set of subtomograms to be represented")
        form.addParam('inputReference', PointerParam, pointerClass="SubTomogram",
                      label='Reference', help="Reference (subtomogram average)")
        form.addParam('inputTomogram', PointerParam, pointerClass="Tomogram",
                      label='Original tomogram', help="Original tomogram from which the subtomograms were extracted")
        form.addParam('paintingType', EnumParam,
                      choices=['Copy','Average','Highlight','Binarize'],
                      default=0, important=True,
                      display=EnumParam.DISPLAY_HLIST,
                      label='Painting mode',
                      help="""The program has several 'painting' options:
                      *Copy*: Copying the reference onto the tomogram
                      *Average*: Setting the region occupied by the reference in the tomogram to the average value of that region
                      *Highlight*: Add the reference multiplied by a constant to the location specified
                      *Binarize*: Copy a binarized version of the reference onto the tomogram'""")
        form.addParam('removeBackground', BooleanParam, default=False, label='Remove background',
                      help= "Set tomogram to 0", condition="paintingType == 0 or paintingType == 3")
        form.addParam('threshold', FloatParam, default=0.5, label='Threshold',
                      help= "threshold applied to tomogram", condition="paintingType == 1 or paintingType == 3")
        form.addParam('constant', FloatParam, default=2, label='Constant',
                      help="constant to multiply the reference",
                      condition="paintingType == 2")

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInput',self.inputSubtomograms.getObjId(),self.inputReference.getObjId(),
                                 self.inputTomogram.getObjId())
        self._insertFunctionStep('runMapBack')
        self._insertFunctionStep('createOutput')

    #--------------------------- STEPS functions -------------------------------
    def convertInput(self,objIdSubtomograms,objIdRef, objIdTomo):
        img = ImageHandler()
        fnTomo = self._getExtraPath('tomogram.mrc')
        img.convert(self.inputTomogram.get(), fnTomo)
        fnRef = self._getExtraPath('reference.mrc')
        img.convert(self.inputReference.get(), fnRef)
        if self.paintingType.get() == 0 or self.paintingType.get() == 3:
            if self.removeBackground.get() == True:
                self.runJob("xmipp_image_operate"," -i %s  --mult 0"%fnTomo)

    def runMapBack(self):
        TsSubtomo = self.inputSubtomograms.get().getSamplingRate()
        TsTomo = self.inputTomogram.get().getSamplingRate()
        scaleFactor = TsSubtomo/TsTomo

        mdGeometry = xmippLib.MetaData()
        for subtomo in self.inputSubtomograms.get():
            objId = mdGeometry.addObject()
            mdGeometry.setValue(xmippLib.MDL_XCOOR,int(subtomo.getCoordinate3D().getX()*scaleFactor),objId)
            mdGeometry.setValue(xmippLib.MDL_YCOOR,int(subtomo.getCoordinate3D().getY()*scaleFactor),objId)
            mdGeometry.setValue(xmippLib.MDL_ZCOOR,int(subtomo.getCoordinate3D().getZ()*scaleFactor),objId)
        fnGeometry = self._getExtraPath("geometry.xmd")
        mdGeometry.write(fnGeometry)

        if self.paintingType.get() == 0:
            painting = 'copy'
        elif self.paintingType.get() == 1:
            painting = 'avg %d' % self.threshold.get()
        elif self.paintingType.get() == 2:
            painting = 'highlight %d' % self.constant.get()
        elif self.paintingType.get() == 3:
            painting = 'copy_binary %d' % self.threshold.get()

       # coordinates = self.inputSubtomograms.getCoordinates3D() # original ones?
        self.runJob("xmipp_tomo_map_back"," -i %s -o %s --geom %s --ref %s --method %s" % (self._getExtraPath("tomogram.mrc"),
                                                                         self._getExtraPath("tomogram_mapped_back.mrc"),
                                                                         self._getExtraPath("geometry.xmd"),
                                                                         self._getExtraPath("reference.mrc"),painting))

    def createOutput(self):
        pass
        # img = ImageHandler()
        # outputTomo = self._createSetOfVolumes()
        # fnOut = self._getExtraPath("tomogram_mapped_back.mrc")
        # img.convert(fnOut,outputTomo)
        # self._defineOutputs(outputTomogram=outputTomo)
        # self._defineSourceRelation(self.inputTomogram, outputTomo)
        # self._defineSourceRelation(self.inputSubtomograms, outputTomo)
        # self._defineSourceRelation(self.inputReference, outputTomo)

    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        summary.append(" ")
        return summary

    def _methods(self):
        methods = []
        methods.append(" ")
        return methods