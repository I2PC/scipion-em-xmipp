# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *           Ruben Sanchez (rsanchez@cnb.csic.es)
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

import sys, shutil
import numpy as np
import mrcfile
import pywt
import pywt.data
from pyworkflow.protocol.params import MultiPointerParam
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume


class XmippProtVolConsensus(ProtInitialVolume):
    """ This protocol performs a fusion of all the input volumes, which should be preprocessed with protocol
    "volume substraction" saving volume 2, in order to be as similar as possible before the fusion. """

    _label = 'volume consensus'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('vols', MultiPointerParam, pointerClass='Volume', label="Volumes",
                      help='Select the volumes for the consensus.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('fusionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ---------------------------------------------------
    def fusionStep(self):
        vol1 = self.vols[0].get()
        vol2 = self.vols[1].get()
        vol1Fn = vol1.getFileName()
        vol2Fn = vol2.getFileName()
        # outVolData = self.computeVolumeConsensus(self.loadVol(vol1Fn), self.loadVol(vol2.getFileName()))
        outVolFn = self._getExtraPath("consensus_volume.mrc")
        # self.saveVol(outVolData, outVolFn, vol1Fn)

        args = " --i1 %s --i2 %s -o %s" % (vol1Fn, vol2Fn, outVolFn)
        self.runJob("xmipp_volume_consensus", args)

    def createOutputStep(self):
        outVol = Volume()
        outVol.setSamplingRate(self.vols[0].get().getSamplingRate())
        outVol.setFileName(self._getExtraPath("consensus_volume.mrc"))
        # origin = Transform()
        # ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
        # shifts = ccp4header.getOrigin()
        # origin.setShiftsTuple(shifts)
        # volume.setOrigin(origin)
        self._defineOutputs(outputVolume=outVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            summary.append("Volume 1: %s\nVolume 2: %s\n Sampling rate: %f A/px" %
                           (self.vols[0].get().getFileName(), self.vols[1].get().getFileName(),
                            self.vols[0].get().getSamplingRate()))
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("Consensus of volumes %s and %s" % (self.vols[0].get().getFileName(),
                                                               self.vols[1].get().getFileName()))
        return methods

    # --------------------------- UTLIS functions --------------------------------------------
    # def computeVolumeConsensus(self, vol1, vol2, wavelet='sym11'):
    #     coeffDict1 = pywt.dwtn(vol1, wavelet)
    #     coeffDict2 = pywt.dwtn(vol2, wavelet)
    #     newDict={}
    #     for key in coeffDict1:
    #         newDict[key] = np.where(np.abs(coeffDict1[key]) > np.abs(coeffDict2[key]), coeffDict1[key], coeffDict2[key])
    #     consensus = pywt.idwtn(newDict, wavelet)
    #     return consensus
    #
    # def saveVol(self, data, fname, fnameToCopyHeader):
    #     shutil.copyfile(fnameToCopyHeader, fname)
    #     with mrcfile.open(fname, "r+", permissive=True) as f:
    #         f.data[:] = data
    #
    # def loadVol(self, fname):
    #     with mrcfile.open(fname, permissive=True) as f:
    #         return f.data.astype(np.float32)
