# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D
from tomo.objects import SetOfTiltSeries
import pwem.emlib.metadata as md
from pwem.emlib import lib

OUTPUT_3DFSC = '3dFSC.mrc'
OUTPUT_DIRECTIONAL_FILTER = 'filteredMap.mrc'
OUTPUT_DIRECTIONAL_DISTRIBUTION = 'Resolution_Distribution.xmd'


class XmippProtSNR(ProtAnalysis3D):
    """    
        blablabla
    """
    _label = 'estimate SNR'
    _lastUpdateVersion = VERSION_2_0
    _devStatus = NEW

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputType', BooleanParam, default=True,
                      label="Is the input a set of Tilt series?",
                      help='True - The input is a set of tilt series with odd-even information.'
                           'False - The input is a set of micrographs.')

        form.addParam('inputTiltSeries', PointerParam, pointerClass='SetOfTiltSeries',
                      label="Tilt Series", important=True, condition='inputType == True',
                      help='Select the first set of micrographs')

        form.addParam('half1', PointerParam, pointerClass='SetOfMicrographs, SetOfTiltSeries',
                      label="Half 1", important=True, condition='inputType == False',
                      help='Select the first set of micrographs')

        form.addParam('half2', PointerParam, pointerClass='SetOfMicrographs, SetOfTiltSeries',
                      label="Half 2", important=True, condition='inputType == False',
                      help='Select the second set of micrographs')

        form.addParam('normalize', BooleanParam, default=True,
                      label="Normalize?",
                      help='True - Adjusts the gray level to zero mean and standard deviation unit.')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.SNRStep)
        #self._insertFunctionStep('createOutputStep')

    def SNRStep(self):
        fnts = 'images.xmd'
        if self.inputType.get():
            for ts in self.inputTiltSeries.get():
                mdts = lib.MetaData()
                tsid = ts.getTsId()

                for index, item in enumerate(ts):
                    tiIndex = item.getLocation()[0]
                    fn = str(tiIndex) + "@" + item.getFileName()
                    nRow = md.Row()
                    nRow.setValue(lib.MDL_IMAGE, fn)
                    if ts.hasOddEven():
                        fnOdd = item.getOdd()
                        fnEven = item.getEven()
                        nRow.setValue(lib.MDL_HALF1, fnOdd)
                        nRow.setValue(lib.MDL_HALF2, fnEven)
                    nRow.setValue(lib.MDL_TSID, tsid)
                    tilt = item.getTiltAngle()
                    nRow.setValue(lib.MDL_ANGLE_TILT, tilt)
                    nRow.addToMd(mdts)

                fnts = os.path.join(self._getExtraPath(), "%s_ts.xmd" % tsid)

                mdts.write(fnts)
                self.vol1Fn = fnts
                fndir = self._getExtraPath("fsc")

                os.mkdir(fndir)

                params = ' -i "%s"' % self.vol1Fn
        else:
            mdts = lib.MetaData()
            for mic in self.half1.get():
                micFn = mic.getFileName()
                nRow = md.Row()
                nRow.setValue(lib.MDL_IMAGE, micFn)
                nRow.addToMd(mdts)
            mdts.write(fnts)

            self.vol1Fn = fnts
            fndir = self._getExtraPath("fsc")

            os.mkdir(fndir)

            params = ' -i "%s"' % self.vol1Fn
            params += ' --half2 "%s"' % self.vol2Fn
        params += ' -o %s' % self._getExtraPath()
        params += ' --sampling %f' % 1.0
        if self.normalize.get():
            params += ' --normalize '
        params += ' --threads %s' % self.numberOfThreads.get()

        self.runJob('xmipp_image_SNR', params)

    # --------------------------- INFO functions ------------------------------
    def _methods(self):
        messages = []
        messages.append('No information to show')
        return messages

    '''
    def _validate(self):
        errors = []

	if not self.inputType.get():
            if not self.half1.get():
                errors.append("You need to select the half1")
            if not self.half2.get():
                errors.append("You need to select the half2")

        return errors
    '''

    def _summary(self):
        summary = []
        summary.append(" ")
        return summary

