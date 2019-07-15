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
from pyworkflow.protocol.params import PointerParam, EnumParam
from pyworkflow.em.convert import ImageHandler
import xmippLib

class XmippProtSubtomoMapBack(EMProtocol):
    """ This protocol maps the subtomogram reference back into the original tomogram volume """

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
                      help="""*Subtomograms*: The subtomograms will appear without background
                      *Highlight*: The subtomograms will appear highlighted on the original tomogram
                      *Binarize*: The subtomograms will appear binarized on the original tomogram
                      *Remove*: The subtomograms will appear removed on the original tomogram'""")

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
            self.runJob("xmipp_image_operate"," -i %s  --mult 0"%fnTomo)

    def runMapBack(self):
        mdGeometry = xmippLib.MetaData()
        for subtomo in self.inputSubtomograms.get():
            objId = mdGeometry.addObject()
            mdGeometry.setValue(xmippLib.MDL_XCOOR,subtomo.getCoordinate3D().getX(),objId)
            mdGeometry.setValue(xmippLib.MDL_YCOOR,subtomo.getCoordinate3D().getY(),objId)
            mdGeometry.setValue(xmippLib.MDL_ZCOOR,subtomo.getCoordinate3D().getZ(),objId)
        fnGeometry = self._getExtraPath("geometry.xmd")
        mdGeometry.write(fnGeometry)

        if self.paintingType.get() == 0:
            painting = 'copy'
        elif self.paintingType.get() == 1:
            painting = 'avg 0.5'
        elif self.paintingType.get() == 2:
            painting = 'highlight 2'
        elif self.paintingType.get() == 3:
            painting = 'copy_binary 0.5'

       # coordinates = self.inputSubtomograms.getCoordinates3D() # original ones?
        self.runJob("xmipp_tomo_map_back"," -i %s -o %s --geom %s --ref %s --method %s" % (self._getExtraPath("tomogram.mrc"),
                                                                         self._getExtraPath("tomogram_mapped_back.mrc"),
                                                                         self._getExtraPath("geometry.xmd"),
                                                                         self._getExtraPath("reference.mrc"),painting))
        aaaaa

    def createOutput(self):
        pass